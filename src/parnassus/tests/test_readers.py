import numpy as np
import pytest

from parnassus.configs.data import DatasetConfig
from parnassus.data import HepMCDataset, RootDataset
from parnassus.utils.mock import get_mock_hepmc_file, get_mock_root_file, get_mock_transforms


@pytest.fixture
def hepmc_fname():
    return get_mock_hepmc_file()


@pytest.fixture
def root_fname():
    return get_mock_root_file()


def test_reader_config_no_file():
    fname = "no_file_test.root"
    with pytest.raises(FileNotFoundError):
        _ = DatasetConfig(file_path=fname)


def test_reader_config_no_batch_size(root_fname: str):
    with pytest.raises(ValueError, match="Asked for batch_loading, but batch_size is not provided"):
        _ = DatasetConfig(file_path=root_fname, batch_loading=True)


def test_root_reader_load_data(root_fname: str):
    var_transform_dict = get_mock_transforms()
    cfg = DatasetConfig(file_path=root_fname, num_events=500)
    _ = RootDataset(cfg, var_transform_dict=var_transform_dict)


def test_root_reader_get_data(root_fname: str):
    var_transform_dict = get_mock_transforms()
    cfg = DatasetConfig(file_path=root_fname, num_events=500)
    reader = RootDataset(cfg, var_transform_dict=var_transform_dict)
    output = reader[0]

    assert "truth_data" in output
    assert "truth_mask" in output
    assert "event_data" in output


def test_hepmc_reader_load_data(hepmc_fname: str):
    var_transform_dict = get_mock_transforms()
    cfg = DatasetConfig(file_path=hepmc_fname, num_events=500)
    _ = HepMCDataset(cfg, var_transform_dict=var_transform_dict)


def test_hepmc_reader_get_data(hepmc_fname: str):
    var_transform_dict = get_mock_transforms()
    cfg = DatasetConfig(file_path=hepmc_fname, num_events=500)
    reader = HepMCDataset(cfg, var_transform_dict=var_transform_dict)
    output = reader[0]

    assert "truth_data" in output
    assert "truth_mask" in output
    assert "event_data" in output


def test_hepmc_root_readers_equivalence(root_fname: str, hepmc_fname: str):
    rng = np.random.default_rng(42)
    var_transform_dict = get_mock_transforms()

    hepmc_cfg = DatasetConfig(file_path=hepmc_fname, num_events=500)
    hepmc_reader = HepMCDataset(hepmc_cfg, var_transform_dict=var_transform_dict)

    root_cfg = DatasetConfig(file_path=root_fname, num_events=500)
    root_reader = RootDataset(root_cfg, var_transform_dict=var_transform_dict)

    for key in ["ht", "met_x", "met_y"]:
        np.testing.assert_allclose(
            hepmc_reader.full_data_array[key],
            root_reader.full_data_array[key],
            atol=1e-5,
            err_msg=f"Error for {key} variable",
        )

    np.testing.assert_allclose(hepmc_reader.truth_cumsum, root_reader.truth_cumsum)

    for i in rng.integers(0, 500, 5):
        hepmc_output = hepmc_reader[i]
        root_output = root_reader[i]

        for key in hepmc_output:
            np.testing.assert_allclose(
                hepmc_output[key], root_output[key], atol=1e-6, err_msg=f"Error for {key}"
            )
