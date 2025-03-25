from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from parnassus.configs import Config
from parnassus.data import HepMCDataset, RootDataset
from parnassus.data.scheme import GenEvent, GenParticleCollection
from parnassus.nn import EulerSampler, ModelWrapper
from parnassus.utils import VarTransform, reshape_phi
from parnassus.utils.logger import ProgressBar, setup_logger, update_task


def generate(config: Config) -> list[GenEvent]:
    log = setup_logger()
    model_config = config.model
    dataset_config = config.dataset_config
    var_transform_dict = {
        key: VarTransform(value) for key, value in model_config.var_transform_dict.items()
    }
    log.info("[green]Starting loading input data...")
    input_file = dataset_config.file_path
    assert isinstance(input_file, Path)
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Trying to load file {input_file}, no file exist!")

    if input_file.suffix == ".root":
        dataset = RootDataset(dataset_config, var_transform_dict=var_transform_dict)
    elif input_file.suffix == ".hepmc":
        dataset = HepMCDataset(dataset_config, var_transform_dict=var_transform_dict)
    else:
        raise ValueError(
            f"Only ROOT or HepMC files are supported as input, got {dataset_config.file_path}"
        )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=4)
    log.info("[green]Data loading completed.")
    log.info("[green]Loading networks...")
    device = torch.device(config.device)
    particle_model = ModelWrapper(model_config.particle_model_path).to(device)
    event_model = ModelWrapper(model_config.event_model_path).to(device)
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
    l_pf_data: dict[str, npt.NDArray[np.float32]] = {
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

            pf = part_sampler.sample(
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
            tr_data_: npt.NDArray[np.float32 | np.int32]
            pf_data_: npt.NDArray[np.float32 | np.int32]
            for j, var in enumerate(dataset_config.truth_variables):
                var_name = var.replace("pt", "ptrel")
                if var_name == "class":
                    tr_data_ = truth_data[..., j:].argmax(-1).cpu().numpy().astype(np.int32)
                    pf_data_ = pf[..., j:].argmax(-1).cpu().numpy().astype(np.int32)
                else:
                    tr_data_ = (
                        var_transform_dict[var_name]
                        .inverse_transform(truth_data[..., j])
                        .numpy()
                        .astype(np.float32)
                    )
                    pf_data_ = (
                        var_transform_dict[var_name]
                        .inverse_transform(pf[..., j])
                        .numpy()
                        .astype(np.float32)
                    )
                    if var_name == "phi":
                        tr_data_ = reshape_phi(tr_data_)
                        pf_data_ = reshape_phi(pf_data_)
                    if var_name == "ptrel":
                        tr_data_ = tr_data_ * truth_ht.reshape(-1, 1)
                        pf_data_ = pf_data_ * pflow_ht.reshape(-1, 1)
                        var_name = "pt"
                l_tr_data[var_name][n : truth_data.shape[0] + n] = tr_data_
                l_pf_data[var_name][n : truth_data.shape[0] + n] = pf_data_
            l_tr_data["ind"][n : truth_data.shape[0] + n] = sample_mask[..., 0].cpu().numpy()
            l_pf_data["ind"][n : truth_data.shape[0] + n] = sample_mask[..., 1].cpu().numpy()
            l_eventNumber[n : truth_data.shape[0] + n] = event_number
            n += truth_data.shape[0]
            progress_bar.update(total_gen_task, advance=1)

    for key in l_pf_data:
        l_pf_data[key] = l_pf_data[key][:n]
        l_tr_data[key] = l_tr_data[key][:n]
    l_eventNumber = l_eventNumber[:n]
    log.info(f"[green]Generated {n} events from requested {len(dataset)}.")
    event_list: list[GenEvent] = []
    with ProgressBar() as progress:
        conv_task = progress.add_task("[green]Converting events.", total=n)
        for i in range(n):
            truth_ind = l_tr_data["ind"][i] > 0
            truth_particles = GenParticleCollection(
                name="truth",
                pt=l_tr_data["pt"][i][truth_ind],
                eta=l_tr_data["eta"][i][truth_ind],
                phi=l_tr_data["phi"][i][truth_ind],
                vx=l_tr_data["vx"][i][truth_ind],
                vy=l_tr_data["vy"][i][truth_ind],
                vz=l_tr_data["vz"][i][truth_ind],
                class_id=l_tr_data["class"][i][truth_ind].astype(np.int32),
            )
            pflow_ind = l_pf_data["ind"][i] > 0
            pflow_particles = GenParticleCollection(
                name="pflow",
                pt=l_pf_data["pt"][i][pflow_ind],
                eta=l_pf_data["eta"][i][pflow_ind],
                phi=l_pf_data["phi"][i][pflow_ind],
                vx=l_pf_data["vx"][i][pflow_ind],
                vy=l_pf_data["vy"][i][pflow_ind],
                vz=l_pf_data["vz"][i][pflow_ind],
                class_id=l_pf_data["class"][i][pflow_ind].astype(np.int32),
            )
            event_list.append(
                GenEvent(
                    event_number=l_eventNumber[i],
                    truth_particles=truth_particles,
                    pflow_particles=pflow_particles,
                )
            )
            progress.update(conv_task, advance=1)
    return event_list
