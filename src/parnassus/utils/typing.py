import numpy as np
import numpy.typing as npt
from torch import Tensor

type TensorDict = dict[str, Tensor]

type FloatArray = npt.NDArray[np.float32]
type IntArray = npt.NDArray[np.int32]
type LongArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]
type VarNameTuple = tuple[str, ...]

__all__ = ["BoolArray", "FloatArray", "IntArray", "LongArray", "TensorDict", "VarNameTuple"]
