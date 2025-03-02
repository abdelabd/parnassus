from particle import PDGID

from .transform import VarTransform, VarTransformConfig


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
