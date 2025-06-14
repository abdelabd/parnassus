import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float32]
type IntArray = npt.NDArray[np.int32]
type LongArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool]

__all__ = ["BoolArray", "FloatArray", "IntArray", "LongArray"]
