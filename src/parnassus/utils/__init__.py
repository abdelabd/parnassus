"""Utility functions and classes for Parnassus."""

import numpy as np
from particle import PDGID

from .transform import TransformRegistry, Unscaler, VarTransform, VarTransformConfig
from .typing import FloatArray, IntArray


def reshape_phi(phi: FloatArray | IntArray) -> FloatArray:
    """Reshape phi angles to be within the range [-pi, pi].

    Parameters
    ----------
    phi : FloatArray | IntArray
        Input array of phi angles.

    Returns
    -------
    FloatArray
        Reshaped array of phi angles within [-pi, pi].
    """
    return np.arctan2(np.sin(phi), np.cos(phi))


def pid_to_class(pid: int) -> int:
    if abs(pid) == 11:
        return 1
    if abs(pid) == 13:
        return 2
    if pid == 0:  # neutral-hadron calorimeter tower (Delphes convention)
        return 3
    p = PDGID(pid)
    if p.is_hadron:
        if p.charge != 0:
            return 0
        return 3
    return 4


def pid_to_class_vectorized(pid: IntArray) -> IntArray:
    """Vectorized Delphes/PDG pid -> class id (0..4), the inverse direction of
    :func:`class_to_pid`.

    Each distinct pid is mapped once through the scalar :func:`pid_to_class`,
    then gathered back to the original shape. This reuses the single tested
    mapping that handles the Delphes neutral-hadron 0 marker and arbitrary PDG
    hadron codes via :class:`PDGID`, and matches the conversion
    :mod:`generate_pseudodata` uses to fill ``pflow_class`` / ``truth_class``.
    """
    pid = np.asarray(pid)
    # ravel keeps inv 1-D across numpy versions; reshape restores the shape.
    uniq, inv = np.unique(pid.ravel(), return_inverse=True)
    lut = np.array([pid_to_class(int(p)) for p in uniq], dtype=np.int64)
    return lut[inv].reshape(pid.shape)


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


def class_to_pid_vectorized(particle_class: IntArray) -> IntArray:
    pid = np.ones_like(particle_class) * 22
    pid[particle_class == 0] = 211
    pid[particle_class == 1] = 11
    pid[particle_class == 2] = 13
    pid[particle_class == 3] = 111
    return pid


def calculate_dr(
    src_eta: FloatArray, src_phi: FloatArray, dst_eta: FloatArray, dst_phi: FloatArray
) -> FloatArray:
    delta_eta = src_eta - dst_eta
    delta_phi = (src_phi - dst_phi + np.pi) % (2 * np.pi) - np.pi

    return np.sqrt(delta_eta**2 + delta_phi**2)


__all__ = [
    "TransformRegistry",
    "Unscaler",
    "VarTransform",
    "VarTransformConfig",
    "reshape_phi",
]
