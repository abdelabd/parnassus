from pathlib import Path

import pytest
from torch import Tensor

from parnassus.nn.sampler import EulerSampler
from parnassus.nn.wrapper import EventModelWrapper, ParticleModelWrapper
from parnassus.utils.mock import get_mock_input_data, get_mock_model_file


@pytest.fixture
def mock_particle_model():
    return get_mock_model_file(mode="part")


@pytest.fixture
def mock_particle_data():
    return get_mock_input_data(mode="part")


@pytest.fixture
def mock_event_model():
    return get_mock_model_file(mode="evt")


@pytest.fixture
def mock_event_data():
    return get_mock_input_data(mode="evt")


def test_particle_model_wrapper_load(mock_particle_model: str):
    _ = ParticleModelWrapper(mock_particle_model)


def test_particle_model_wrapper_forward(
    mock_particle_model: str, mock_particle_data: dict[str, Tensor]
):
    model = ParticleModelWrapper(mock_particle_model)
    _ = model.forward(
        fs_data=mock_particle_data["fs_data"],
        tr_data=mock_particle_data["tr_data"],
        mask=mock_particle_data["mask"],
        timestep=mock_particle_data["timestep"],
        global_data=mock_particle_data["global_data"],
    )


def test_real_particle_model_wrapper_forward(mock_particle_data: dict[str, Tensor]):
    real_path = (
        Path(__file__).cwd().joinpath("src/parnassus/pretrained_models/exported_part_model.pt2")
    )
    model = ParticleModelWrapper(real_path)
    _ = model.forward(
        fs_data=mock_particle_data["fs_data"],
        tr_data=mock_particle_data["tr_data"],
        mask=mock_particle_data["mask"],
        timestep=mock_particle_data["timestep"],
        global_data=mock_particle_data["global_data"],
    )


def test_event_model_wrapper_load(mock_event_model: str):
    _ = EventModelWrapper(mock_event_model)


def test_event_model_wrapper_forward(mock_event_model: str, mock_event_data: dict[str, Tensor]):
    model = EventModelWrapper(mock_event_model)
    _ = model.forward(
        fs_data=mock_event_data["fs_data"],
        tr_data=mock_event_data["tr_data"],
        mask=mock_event_data["mask"],
        timestep=mock_event_data["timestep"],
        global_data=mock_event_data["global_data"],
    )


def test_real_event_model_wrapper_forward(mock_event_data: dict[str, Tensor]):
    real_path = (
        Path(__file__).cwd().joinpath("src/parnassus/pretrained_models/exported_evt_model.pt2")
    )
    model = EventModelWrapper(real_path)
    _ = model.forward(
        fs_data=mock_event_data["fs_data"],
        tr_data=mock_event_data["tr_data"],
        mask=mock_event_data["mask"],
        timestep=mock_event_data["timestep"],
        global_data=mock_event_data["global_data"],
    )


def test_euler_sampler(mock_particle_model: str, mock_particle_data: dict[str, Tensor]):
    sampler = EulerSampler(n_steps=1, zero_init_padded=False, random_seed=42)
    model = ParticleModelWrapper(mock_particle_model)
    _ = sampler.sample(
        model=model,
        truth=mock_particle_data["tr_data"],
        pflow_shape=mock_particle_data["tr_data"].size(),
        mask=mock_particle_data["mask"],
        global_data=mock_particle_data["global_data"],
    )
