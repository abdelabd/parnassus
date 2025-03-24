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


def class_to_pid(particle_class: int) -> int:
    if particle_class == 0:
        return 211
    if particle_class == 1:
        return 11
    if particle_class == 2:
        return 13
    if particle_class == 3:
        return 111
    return 22


def class_to_pid_vecorized(particle_class: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    pid = np.ones_like(particle_class) * 22
    pid[particle_class == 0] = 211
    pid[particle_class == 1] = 11
    pid[particle_class == 2] = 13
    pid[particle_class == 3] = 111
    return pid


__all__ = ["VarTransform", "VarTransformConfig"]
