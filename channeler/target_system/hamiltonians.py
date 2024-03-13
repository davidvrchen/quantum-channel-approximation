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

import qutip as qt
import numpy as np

from .settings import TargetSystemSettings, DecaySettings, TFIMSettings
from ..utils.pauli_matrices import Id, X, Z


Hamiltonian = qt.Qobj


def decay_hamiltonian(s: DecaySettings) -> Hamiltonian:
    """Decay Hamiltonian def'd by settings ``s``.

    Note: internally this function is still numpy based,
    Todo: switch over to qutip"""

    if s.m == 1:
        (om0,) = s.omegas
        return qt.Qobj(om0 * X, dims=[[2], [2]])

    if s.m == 2:
        om0, om1 = s.omegas
        return qt.Qobj(
            om0 * np.kron(X, Id)
            + om1 * np.kron(Id, X)
            + s.ryd_interaction
            * np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]),
            dims=[[2, 2], [2, 2]],
        )

    if s.m == 3:
        om0, om1, om2 = s.omegas

        return qt.Qobj(
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
            ),
            dims=[[2, 2, 2], [2, 2, 2]],
        )


def tfim_hamiltonian(s: TFIMSettings) -> Hamiltonian:
    """Transverse field Ising model Hamiltonian def'd by settings ``s``.

    Note: internally this function is still numpy based,
    Todo: switch over to qutip
    """

    if s.m == 2:
        return qt.Qobj(
            s.j_en * (np.kron(Z, Id) @ np.kron(Id, Z))
            - s.h_en * (np.kron(X, Id) + np.kron(Id, X)),
            dims=[[2, 2], [2, 2]],
        )

    if s.m == 3:
        return qt.Qobj(
            s.j_en
            * (
                np.kron(np.kron(Z, Id), Id) @ np.kron(np.kron(Id, Z), Id)
                + np.kron(np.kron(Id, Id), Z) @ np.kron(np.kron(Id, Z), Id)
            )
            - s.h_en
            * (
                np.kron(np.kron(X, Id), Id)
                + np.kron(np.kron(Id, X), Id)
                + np.kron(np.kron(Id, Id), X)
            ),
            dims=[[2, 2, 2], [2, 2, 2]],
        )


def create_hamiltonian(s: TargetSystemSettings) -> Hamiltonian:
    """Convenience function that creates the appropriate
    Hamiltonian from settings."""

    if isinstance(s, DecaySettings):
        return decay_hamiltonian(s)

    if isinstance(s, TFIMSettings):
        return tfim_hamiltonian(s)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
