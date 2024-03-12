"""
Helper functions that can construct the Hamiltonian
and the jump operators of the Lindbladian.


References:
    Original code by @lviss

Info:
    Created on Mon Feb 26 2024

    @author: davidvrchen
"""
import numpy as np

from ..channeler.utils.pauli_matrices import Id, X, Y, Z
from ..channeler.utils.settings import GeneralSettings


def jump_operators(s: GeneralSettings) -> np.ndarray:
    """Construct jump operators needed for the Lindbladian from settings."""

    m = s.m
    gam0 = s.gam0
    gam1 = s.gam1
    gam2 = s.gam2

    if m == 1:
        y0 = gam0 ** (1 / 2)

        # |1> to |0>, |0> to |0
        A0 = [
            [0, y0],
            [0, 0],
        ]

        return np.array([A0])

    if m == 2:
        y0 = gam0 ** (1 / 2)
        y1 = gam1 ** (1 / 2)

        # |. 1> to |. 0>
        A0 = [
            [0, y0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, y0],
            [0, 0, 0, 0],
        ]

        # |1 .> to |0 .>
        A1 = [
            [0, 0, y1, 0],
            [0, 0, 0, y1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        return np.array([A0, A1])

    if m == 3:
        y0 = gam0 ** (1 / 2)
        y1 = gam1 ** (1 / 2)
        y2 = gam2 ** (1 / 2)

        A0 = [
            [0, y0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, y0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, y0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, y0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]

        A1 = [
            [0, 0, y1, 0, 0, 0, 0, 0],
            [0, 0, 0, y1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, y1, 0],
            [0, 0, 0, 0, 0, 0, 0, y1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]

        A2 = [
            [0, 0, 0, 0, y2, 0, 0, 0],
            [0, 0, 0, 0, 0, y2, 0, 0],
            [0, 0, 0, 0, 0, 0, y2, 0],
            [0, 0, 0, 0, 0, 0, 0, y2],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]

        return np.array([A0, A1, A2])


def hamiltonian(s: GeneralSettings) -> np.ndarray:
    """Creates the Hamiltonian for the specified target system
    to be modeled from settings.
    """
    lb_type = s.lb_type

    if lb_type == "decay":
        return decay_hamiltonian(
            m=s.m,
            om0=s.lb_settings.om0,
            om1=s.lb_settings.om1,
            om2=s.lb_settings.om2,
            ryd_interaction=s.lb_settings.ryd_interaction,
        )

    if lb_type == "tfim":
        return tfim_hamiltonian(m=s.m, j_en=s.lb_settings.j_en, h_en=s.lb_settings.h_en)
    



def decay_hamiltonian(
    m: int, ryd_interaction: float, om0: float, om1: float = None, om2: float = None
):
    """Decay Hamiltonian for n_qubits

    Parameters:
    -----------

    m: number of qubits (only 1, 2 or 3 supported)

    om0:
    """

    if m == 1:
        return om0 * X

    if m == 2:
        return (
            om0 * np.kron(X, Id)
            + om1 * np.kron(Id, X)
            + ryd_interaction
            * np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        )

    if m == 3:
        return (
            om0 * np.kron(np.kron(X, Id), Id)
            + om1 * np.kron(np.kron(Id, X), Id)
            + om2 * np.kron(np.kron(Id, Id), X)
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
            )
        )


def tfim_hamiltonian(m: int, j_en, h_en):
    """Transverse field Ising model Hamiltonian for n_qubits.

    Parameters:
    -----------

    m: number of qubits (only 2 or 3 supported)

    j_en: interaction strength between neighbouring spins

    h_en: strength of transverse magnetic field
    """

    if m == 2:
        return j_en * (np.kron(Z, Id) @ np.kron(Id, Z)) - h_en * (
            np.kron(X, Id) + np.kron(Id, X)
        )

    if m == 3:
        return j_en * (
            np.kron(np.kron(Z, Id), Id) @ np.kron(np.kron(Id, Z), Id)
            + np.kron(np.kron(Id, Id), Z) @ np.kron(np.kron(Id, Z), Id)
        ) - h_en * (
            np.kron(np.kron(X, Id), Id)
            + np.kron(np.kron(Id, X), Id)
            + np.kron(np.kron(Id, Id), X)
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
