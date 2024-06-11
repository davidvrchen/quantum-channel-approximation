"""
Some function to create observables.
"""

import qutip as qt

from q_channel_approx.pauli_strings import (
    k_random_pauli_strs,
    order_n_pauli_strs,
    all_pauli_strs,
    pauli_strs_2_ops,
)


def k_random_observables(m: int, k: int, seed: int) -> list[qt.Qobj]:
    """Generate `k` random observables on `m` qubits.

    Args:
        m (int): _description_
        k (int): number of observables
        seed (int): seed used to generate the Pauli strings

    Returns:
        list[qt.Qobj]: list of the observables
    """

    pauli_strs = k_random_pauli_strs(m=m, k=k, seed=seed)
    return pauli_strs_2_ops(pauli_strs)


def order_n_observables(m: int, n: int) -> list[qt.Qobj]:
    """Generate all observables on `m` qubits upto order `n`.

    Args:
        m (int): number of qubits.
        n (int): highest order Pauli strings included.

    Returns:
        list[qt.Qobj]: list of the observables.
    """

    pauli_strs = order_n_pauli_strs(m=m, n=n)
    return pauli_strs_2_ops(pauli_strs)


def all_observables(m: int) -> list[qt.Qobj]:
    """All observables on `m` qubits.

    Args:
        m (int): number of qubits.

    Returns:
        list[qt.Qobj]: list of all observables.
    """

    pauli_strs = all_pauli_strs(m=m)
    return pauli_strs_2_ops(pauli_strs)
