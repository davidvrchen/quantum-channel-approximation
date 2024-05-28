"""
Provides functions to construct the Hamiltonians for the various target systems as
defined in target_systems.py
Supported target systems:
    decay
    tfim


References:
    original code for decay_examply by @lviss

Note:
    Current implementation mix between numpy as qutip,
    to be converted to qutip

Info:
    Created on Mon March 11 2024

    Last update on Thu Apr 4 2024

    @author: davidvrchen
"""

import qutip as qt
import numpy as np

from .target_systems import TargetSystem, DecaySystem, TFIMSystem
from .pauli_spin_matrices import Idnp, Xnp, Znp, X, Z, Id


def decay_hamiltonian(m: int, omegas: tuple[float], ryd_interaction: float):
    """to be added"""

    if m == 1:
        (om0,) = omegas
        return qt.Qobj(om0 * X, dims=[[2], [2]])

    if m == 2:
        om0, om1 = omegas
        return qt.Qobj(
            om0 * np.kron(Xnp, Idnp)
            + om1 * np.kron(Idnp, Xnp)
            + ryd_interaction
            * np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                ]
            ),
            dims=[[2, 2], [2, 2]],
        )

    if m == 3:
        om0, om1, om2 = omegas

        return qt.Qobj(
            om0 * np.kron(np.kron(Xnp, Idnp), Idnp)
            + om1 * np.kron(np.kron(Idnp, Xnp), Idnp)
            + om2 * np.kron(np.kron(Idnp, Idnp), Xnp)
            + ryd_interaction
            * np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 2],
                ]
            ),
            dims=[[2, 2, 2], [2, 2, 2]],
        )


def tfim_hamiltonian(m: int, j_en: float, h_en: float):
    """to be added"""

    if m == 2:
        return qt.Qobj(
            j_en * (np.kron(Znp, Idnp) @ np.kron(Idnp, Znp))
            - h_en * (np.kron(Xnp, Idnp) + np.kron(Idnp, Xnp)),
            dims=[[2, 2], [2, 2]],
        )

    if m == 3:
        return qt.Qobj(
            j_en
            * (
                np.kron(np.kron(Znp, Idnp), Idnp) @ np.kron(np.kron(Idnp, Znp), Idnp)
                + np.kron(np.kron(Idnp, Idnp), Znp) @ np.kron(np.kron(Idnp, Znp), Idnp)
            )
            - h_en
            * (
                np.kron(np.kron(Xnp, Idnp), Idnp)
                + np.kron(np.kron(Idnp, Xnp), Idnp)
                + np.kron(np.kron(Idnp, Idnp), Xnp)
            ),
            dims=[[2, 2, 2], [2, 2, 2]],
        )


def _tfim_hamiltonian(s: TFIMSystem):
    """to be added"""
    # read settings from TFIMSystem
    m = s.m
    j_en = s.j_en
    h_en = s.h_en

    return tfim_hamiltonian(m=m, j_en=j_en, h_en=h_en)


def _decay_hamiltonian(s: DecaySystem):
    """Convenience function to create Hamiltonian from DecaySystem object."""
    # read settings from DecaySystem
    m = s.m
    omegas = s.omegas
    ryd_interaction = s.ryd_interaction

    return decay_hamiltonian(m=m, omegas=omegas, ryd_interaction=ryd_interaction)


def create_hamiltonian(s: TargetSystem):
    """Convenience function that creates the appropriate
    Hamiltonian from settings."""

    if isinstance(s, DecaySystem):
        return _decay_hamiltonian(s)

    if isinstance(s, TFIMSystem):
        return _tfim_hamiltonian(s)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
