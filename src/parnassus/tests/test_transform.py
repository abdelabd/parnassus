import re

import pytest

from parnassus.utils.transform import TRANSFORM_FUNCTIONS, TRANSFORM_TYPES, VarTransformConfig


def get_default_params() -> dict[str, str | bool | float | None]:
    return {
        "name": "pt",
        "transform_type": "std",
        "transform_fn": "log",
        "mean": 0,
        "std": 1,
        "min": 0,
        "max": 1,
    }


def test_var_transform_confg_fn():
    params = get_default_params()
    params["transform_fn"] = "exp"
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Expected transform_fn for var {params['name']} "
            f"be in {TRANSFORM_FUNCTIONS}, got {params['transform_fn']}"
        ),
    ):
        _ = VarTransformConfig(**params)  # pyright: ignore[reportArgumentType]


def test_var_transform_wrong_type():
    params = get_default_params()
    params["transform_type"] = "max"
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Expected transform_type for var {params['name']} "
            f"be in {TRANSFORM_TYPES}, got {params['transform_type']}"
        ),
    ):
        _ = VarTransformConfig(**params)  # pyright: ignore[reportArgumentType]


def test_var_transform_std_type():
    params = get_default_params()
    params["transform_type"] = "std"
    params["mean"] = None
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"For var {params['name']} and 'std' transform_type mean and std values "
            f"should be provided, got mean={params['mean']}, std={params['std']}"
        ),
    ):
        _ = VarTransformConfig(**params)  # pyright: ignore[reportArgumentType]


def test_var_transform_minmax_type():
    params = get_default_params()
    params["transform_type"] = "min_max"
    params["min"] = None
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"For var {params['name']} and 'min_max' transform_type min and max values "
            f"should be provided, got min={params['min']}, max={params['max']}"
        ),
    ):
        _ = VarTransformConfig(**params)  # pyright: ignore[reportArgumentType]
