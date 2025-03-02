import argparse
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import awkward as ak
import numpy as np
import numpy.typing as npt
import torch
import uproot
import yaml
from rich.progress import Progress, TaskID
from torch import Tensor
from torch.utils.data import DataLoader

from parnassus.data import DatasetConfig, HepMCDataset, RootDataset
from parnassus.nn import EulerSampler, EventModelWrapper, ParticleModelWrapper
from parnassus.utils import VarTransform, VarTransformConfig
from parnassus.utils.logger import ProgressBar, setup_logger


def reshape_phi(phi: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.arctan2(np.sin(phi), np.cos(phi))


def parse_args(args: Sequence[str] | None):
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("-n", "--n_steps", type=int, default=50)
    _ = parser.add_argument("-g", "--gpu", type=int, default=0)
    _ = parser.add_argument("-e", "--eval_dir", type=str, default="evals")
    _ = parser.add_argument("-ne", "--num_events", type=int, default=100_00)
    _ = parser.add_argument("-bs", "--batch_size", type=int, default=200)
    _ = parser.add_argument("--test_path", type=str, default=None)
    _ = parser.add_argument("--prefix", type=str, default="")
    return parser.parse_args(args)


TRANSFORM_CONFIG_PATH = (
    Path(__file__).cwd().joinpath("src/parnassus/configs/var_transform_cms.yaml")
)
PART_MODEL_PATH = (
    Path(__file__).cwd().joinpath("src/parnassus/pretrained_models/exported_part_model.pt2")
)
EVENT_MODEL_PATH = (
    Path(__file__).cwd().joinpath("src/parnassus/pretrained_models/exported_evt_model.pt2")
)


def main(args: Sequence[str] | None = None) -> None:
    parsed_args = parse_args(args)
    log = setup_logger()
    title = " Starting processing "
    log.info(f"[bold green]{title:-^100}")
    start = datetime.now()
    log.info(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    with open(TRANSFORM_CONFIG_PATH) as f:
        transform_config = yaml.safe_load(f)
        var_transform_dict = {
            key: VarTransform(
                VarTransformConfig(
                    name=key,
                    min=value["min"],
                    max=value["max"],
                    mean=value["mean"],
                    std=value["std"],
                )
            )
            for key, value in transform_config.items()
        }
    dataset_config = DatasetConfig(
        file_path=Path(parsed_args.test_path),
        num_events=parsed_args.num_events,
    )
    log.info("[green]Starting loading input data...")
    if dataset_config.file_path.suffix == ".root":
        dataset = RootDataset(dataset_config, var_transform_dict=var_transform_dict)
    elif dataset_config.file_path.suffix == ".hepmc":
        dataset = HepMCDataset(dataset_config, var_transform_dict=var_transform_dict)
    else:
        raise ValueError(
            f"Only ROOT or HepMC files are supported as input, got {dataset_config.file_path}"
        )
    dataloader = DataLoader(dataset, batch_size=parsed_args.batch_size, num_workers=4)
    log.info("[green]Data loading completed.")
    log.info("[green]Loading networks...")
    # device = torch.device(f"cuda:{parsed_args.gpu}")
    device = torch.device("mps")
    particle_model = ParticleModelWrapper(PART_MODEL_PATH).to(device)
    event_model = EventModelWrapper(EVENT_MODEL_PATH).to(device)
    log.info("[green]Networks loading completed.")

    n_events = len(dataset)
    l_tr_data: dict[str, npt.NDArray[np.float32]] = {
        key.replace("pflow_", "").replace("ptrel", "pt"): np.zeros(
            (
                n_events,
                dataset_config.max_particles,
            ),
            dtype=np.float32,
        )
        for key in [*dataset_config.truth_variables, "ind"]
    }
    l_fs_data: dict[str, npt.NDArray[np.float32]] = {
        key.replace("pflow_", "").replace("ptrel", "pt"): np.zeros(
            (
                n_events,
                dataset_config.max_particles,
            ),
            dtype=np.float32,
        )
        for key in [*dataset_config.truth_variables, "ind"]
    }
    l_eventNumber = np.zeros(n_events, dtype=np.int32)
    n = 0
    ht_mean, ht_std = (
        var_transform_dict["ht"].cfg.shift,
        var_transform_dict["ht"].cfg.scale,
    )
    raw_n = 0
    evt_sampler = EulerSampler(50)
    part_sampler = EulerSampler(50)

    with ProgressBar() as progress_bar:
        total_gen_task = progress_bar.add_task("[green]Generating data", total=len(dataloader))
        evt_sampler_task = progress_bar.add_task(
            "[green]Sampling event data", total=evt_sampler.n_steps
        )
        part_sampler_task = progress_bar.add_task(
            "[green]Sampling particle data", total=part_sampler.n_steps
        )

        def update_task(progress_bar: Progress, task: TaskID) -> Callable[[], None]:
            def update():
                progress_bar.update(task, advance=1)

            return update

        for batch in dataloader:
            progress_bar.reset(evt_sampler_task)
            progress_bar.reset(part_sampler_task)
            truth_data: Tensor = batch["truth_data"]
            truth_mask: Tensor = batch["truth_mask"]
            global_data: Tensor = batch["event_data"]
            npf_ext_shape = (len(truth_data), 4)
            pred = evt_sampler.sample(
                event_model,
                truth_data.to(device),
                npf_ext_shape,
                truth_mask.to(device),
                global_data=global_data.to(device),
                callback=update_task(progress_bar, evt_sampler_task),
            ).cpu()
            n_pf_pred = var_transform_dict["npart"].inverse_transform(pred[..., 1]).round().int()
            idxs = np.arange(pred.shape[0])
            pf_ht_pred = pred[..., 0]

            good_idxs = idxs[(pf_ht_pred > -ht_mean / ht_std) & (n_pf_pred > 0) & (n_pf_pred < 400)]

            pf_ht_pred[pf_ht_pred < -ht_mean / ht_std] = global_data[..., -1][
                pf_ht_pred < -ht_mean / ht_std
            ]

            n_tr = truth_mask.sum(-1).int()
            n_pf_pred[n_pf_pred < 1] = n_tr[n_pf_pred < 1]
            n_pf_pred[n_pf_pred > 400] = n_tr[n_pf_pred > 400]

            pf_met_x_pred = pred[..., 2]
            pf_met_y_pred = pred[..., 3]

            event_number = dataset.full_data_array["eventNumber"][
                raw_n : truth_data.shape[0] + raw_n
            ]

            truth_data = truth_data[good_idxs]
            truth_mask = truth_mask[good_idxs]
            global_data = global_data[good_idxs]

            n_pf_pred = n_pf_pred[good_idxs]
            pf_ht_pred = pf_ht_pred[good_idxs]
            pf_met_x_pred = pf_met_x_pred[good_idxs]
            pf_met_y_pred = pf_met_y_pred[good_idxs]

            event_number = event_number[good_idxs]

            raw_n += len(idxs)

            sample_mask = torch.zeros(
                (truth_data.shape[0], dataset_config.max_particles, 2), dtype=torch.bool
            )
            sample_mask[..., 0] = truth_mask

            for j in range(n_pf_pred.shape[0]):
                sample_mask[j, : n_pf_pred[j], 1] = True
                sample_mask[j, n_pf_pred[j] :, 1] = False

            truth_ht_scaled = global_data[..., -1]
            global_data = torch.cat(
                [
                    global_data[..., :-4],  # scale info
                    global_data[..., -4:-2],  # truth_met_x, truth_met_y
                    pf_met_x_pred.unsqueeze(-1),  # pf_met_x
                    pf_met_y_pred.unsqueeze(-1),  # pf_met_y
                    global_data[..., -2:],  # truth_npart, truth_ht
                    var_transform_dict["npart"]
                    .transform(n_pf_pred.float())
                    .unsqueeze(-1),  # n_pf_pred
                    pf_ht_pred.unsqueeze(-1),  # pf_ht_pred
                ],
                -1,
            )

            fs = part_sampler.sample(
                particle_model,
                truth_data.to(device),
                (*truth_data.shape[:-1], 11),
                sample_mask.to(device),
                global_data=global_data.to(device),
                callback=update_task(progress_bar, part_sampler_task),
            ).cpu()

            pflow_ht: npt.NDArray[np.float32] = (
                var_transform_dict["ht"].inverse_transform(pf_ht_pred).numpy()
            )
            truth_ht: npt.NDArray[np.float32] = (
                var_transform_dict["ht"].inverse_transform(truth_ht_scaled).numpy()
            )
            tr_data_: npt.NDArray[np.float32]
            fs_data_: npt.NDArray[np.float32]
            for j, var in enumerate(dataset_config.truth_variables):
                var_name = var.replace("pt", "ptrel")
                if var_name == "class":
                    tr_data_ = truth_data[..., j:].argmax(-1).cpu().numpy().astype(np.float32)
                    fs_data_ = fs[..., j:].argmax(-1).cpu().numpy().astype(np.float32)
                else:
                    tr_data_ = (
                        var_transform_dict[var_name]
                        .inverse_transform(truth_data[..., j])
                        .numpy()
                        .astype(np.float32)
                    )
                    fs_data_ = (
                        var_transform_dict[var_name]
                        .inverse_transform(fs[..., j])
                        .numpy()
                        .astype(np.float32)
                    )
                    if var_name == "phi":
                        tr_data_ = reshape_phi(tr_data_)
                        fs_data_ = reshape_phi(fs_data_)
                    if var_name == "ptrel":
                        tr_data_ = tr_data_ * truth_ht.reshape(-1, 1)
                        fs_data_ = fs_data_ * pflow_ht.reshape(-1, 1)
                        var_name = "pt"
                l_tr_data[var_name][n : truth_data.shape[0] + n] = tr_data_
                l_fs_data[var_name][n : truth_data.shape[0] + n] = fs_data_
            l_tr_data["ind"][n : truth_data.shape[0] + n] = sample_mask[..., 0].cpu().numpy()
            l_fs_data["ind"][n : truth_data.shape[0] + n] = sample_mask[..., 1].cpu().numpy()
            l_eventNumber[n : truth_data.shape[0] + n] = event_number
            n += truth_data.shape[0]
            progress_bar.update(total_gen_task, advance=1)

    for key in l_fs_data:
        l_fs_data[key] = l_fs_data[key][:n]
        l_tr_data[key] = l_tr_data[key][:n]
    l_eventNumber = l_eventNumber[:n]

    eval_path = Path(parsed_args.eval_dir).joinpath("output.root")
    log.info(f"[green]Saving generated events to {eval_path.absolute()}")
    with uproot.recreate(eval_path) as file:
        file["evt_tree"] = {
            "pflow": ak.zip({
                key: ak.Array(l_fs_data[key].astype(np.float32).tolist()) for key in l_fs_data
            }),
            "truth": ak.zip({
                key: ak.Array(l_tr_data[key].astype(np.float32).tolist()) for key in l_tr_data
            }),
            "eventNumber": ak.Array(l_eventNumber),
        }

    end = datetime.now()
    title = " Finished processing! "
    log.info(f"[bold green]{title:-^100}")
    log.info(f"End time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Elapsed time: {str(end - start).split('.')[0]}")


if __name__ == "__main__":
    main()
