import numpy as np
import numpy.typing as npt
from particle import PDGID

from .transform import VarTransform, VarTransformConfig


def reshape_phi(phi: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.arctan2(np.sin(phi), np.cos(phi))


def pid_to_class(pid: int) -> int:
    if abs(pid) == 11:
        return 1
    if abs(pid) == 13:
        return 2
    p = PDGID(pid)
    if p.is_hadron:
        if p.charge != 0:
            return 0
        return 3
    return 4


__all__ = ["VarTransform", "VarTransformConfig"]
