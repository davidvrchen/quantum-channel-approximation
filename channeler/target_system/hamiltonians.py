"""
Provides a function to construct the Hamiltonian
of the various target systems.
Supported target systems:
    decay
    tfim


References:
    original code for decay_examply by @lviss

Info:
    Created on Mon March 11 2024

    @author: davidvrchen
"""

import numpy as np

from .settings import DecaySettings, TFIMSettings
from ..utils.pauli_matrices import Id, X, Z


def decay_hamiltonian(s: DecaySettings):
    """Decay Hamiltonian def'd by settings ``s``."""

    if s.m == 1:
        (om0,) = s.omegas
        return om0 * X

    if s.m == 2:
        om0, om1 = s.omegas
        return (
            om0 * np.kron(X, Id)
            + om1 * np.kron(Id, X)
            + s.ryd_interaction
            * np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        )

    if s.m == 3:
        om0, om1, om2 = s.omegas

        return (
            om0 * np.kron(np.kron(X, Id), Id)
            + om1 * np.kron(np.kron(Id, X), Id)
            + om2 * np.kron(np.kron(Id, Id), X)
            + s.ryd_interaction
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
            )
        )


def tfim_hamiltonian(s: TFIMSettings):
    """Transverse field Ising model Hamiltonian def'd by settings ``s``."""

    if s.m == 2:
        return s.j_en * (np.kron(Z, Id) @ np.kron(Id, Z)) - s.h_en * (
            np.kron(X, Id) + np.kron(Id, X)
        )

    if s.m == 3:
        return s.j_en * (
            np.kron(np.kron(Z, Id), Id) @ np.kron(np.kron(Id, Z), Id)
            + np.kron(np.kron(Id, Id), Z) @ np.kron(np.kron(Id, Z), Id)
        ) - s.h_en * (
            np.kron(np.kron(X, Id), Id)
            + np.kron(np.kron(Id, X), Id)
            + np.kron(np.kron(Id, Id), X)
        )
