"""
Provides functions to construct the Hamiltonians for the various target systems as
defined in target_systems.py
"""

import qutip as qt
import numpy as np

from .target_systems import TargetSystem, DecaySystem, TFIMSystem, NothingSystem
from .pauli_spin_matrices import Idnp, Xnp, Znp, X


def I_hamiltonian(m: int) -> qt.Qobj:
    """Identity matrix Hamiltonian, mainly used for testing.

    Args:
        m (int): number of qubits.

    Returns:
        qt.Qobj: Identity matrix Hamiltonian.
    """

    return qt.Qobj(np.identity(2**m))


def decay_hamiltonian(m: int, omegas: tuple[float], ryd_interaction: float) -> qt.Qobj:
    """Hamiltonian for Rabi oscillations of `m` atoms.

    Args:
        m (int): number of qubits.
        omegas (tuple[float]): Rabi frequencies of the atoms.
        ryd_interaction (float): interaction between the atoms.

    Returns:
        qt.Qobj: the corresponding Hamiltonian.
    """

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


def tfim_hamiltonian(m: int, j_en: float, h_en: float) -> qt.Qobj:
    """Hamiltonian of the Transverse field Ising model on `m` qubits.

    Args:
        m (int): number of qubits.
        j_en (float): strength of the interaction between neighboring spins.
        h_en (float): strength of the magnetic field.

    Returns:
        qt.Qobj: The corresponding Hamiltonian.
    """

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


def _I_hamiltonian(s: NothingSystem) -> qt.Qobj:
    m = s.m

    return I_hamiltonian(m=m)


def _tfim_hamiltonian(s: TFIMSystem) -> qt.Qobj:
    """to be added"""
    # read settings from TFIMSystem
    m = s.m
    j_en = s.j_en
    h_en = s.h_en

    return tfim_hamiltonian(m=m, j_en=j_en, h_en=h_en)


def _decay_hamiltonian(s: DecaySystem) -> qt.Qobj:
    """Convenience function to create Hamiltonian from DecaySystem object."""
    # read settings from DecaySystem
    m = s.m
    omegas = s.omegas
    ryd_interaction = s.ryd_interaction

    return decay_hamiltonian(m=m, omegas=omegas, ryd_interaction=ryd_interaction)


def create_hamiltonian(s: TargetSystem) -> qt.Qobj:
    """Convenience function that creates the appropriate
    Hamiltonian from settings."""

    if isinstance(s, DecaySystem):
        return _decay_hamiltonian(s)

    if isinstance(s, TFIMSystem):
        return _tfim_hamiltonian(s)

    if isinstance(s, NothingSystem):
        return _I_hamiltonian(s)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
