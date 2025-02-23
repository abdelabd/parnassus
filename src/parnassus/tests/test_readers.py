import pytest

from parnassus.data import DatasetConfig, RootDataset
from parnassus.utils.mock import get_mock_root_file, get_mock_transforms


def test_reader_config_no_file():
    fname = "no_file_test.root"
    with pytest.raises(FileNotFoundError):
        _ = DatasetConfig(file_path=fname)


def test_reader_config_no_batch_size():
    fname = get_mock_root_file()
    with pytest.raises(ValueError, match="Asked for batch_loading, but batch_size is not provided"):
        _ = DatasetConfig(file_path=fname, batch_loading=True)


def test_root_reader_load_data():
    fname = get_mock_root_file()
    var_transform_dict = get_mock_transforms()
    cfg = DatasetConfig(file_path=fname, num_events=500)
    reader = RootDataset(cfg, var_transform_dict=var_transform_dict)
    reader.load_data()


def test_root_reader_get_data():
    fname = get_mock_root_file()
    var_transform_dict = get_mock_transforms()
    cfg = DatasetConfig(file_path=fname, num_events=500)
    reader = RootDataset(cfg, var_transform_dict=var_transform_dict)
    reader.load_data()
    output = reader[0]

    assert "truth_data" in output
    assert "truth_mask" in output
    assert "event_data" in output
