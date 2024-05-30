"""
Dataclasses that represent possible sets of Observables,
some function to then create those observables and some other .
"""

from dataclasses import dataclass

import qutip as qt

from q_channel_approx.pauli_strings import (
    k_random_pauli_strs,
    order_n_pauli_strs,
    all_pauli_strs,
    pauli_strs_2_ops,
)


@dataclass
class Observables:
    """Dataclass that represents a set of observables on m qubits.

    Args:
    -----
    m (int): number of qubits.
    """

    m: int


@dataclass
class KRandomObservables(Observables):
    """Dataclass that represents a set of k randomly chosen pauli string observables on m qubits.

    Args:
    -----
    m (int): number of qubits.

    k (int): number of pauli string observables to pick

    seed (int, optional): seed used when picking the k pauli strings
    """

    k: int
    seed: int = None


@dataclass
class OrderNObservables(Observables):
    """Dataclass that represents the set of pauli strings observables of at most order n on m qubits.

    Args:
    -----
    m (int): number of qubits.

    n (int): highest order pauli strings to use
    """

    n: int


@dataclass
class AllObservables(Observables):
    """Dataclass that represents the set of all pauli strings observables on m qubits.

    Args:
    -----
    m (int): number of qubits.
    """


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


def _k_random_observables(s: KRandomObservables) -> list[qt.Qobj]:
    """Convenience function to create all pauli strings from KRandomObservables.

    Args:
        s (KRandomObservables): settings object.

    Returns:
        list[qt.Qobj]: list of random observables.
    """

    # read settings
    m = s.m
    k = s.k
    seed = s.seed

    return k_random_observables(m=m, k=k, seed=seed)


def _order_n_observables(s: OrderNObservables) -> list[qt.Qobj]:
    """Convenience function to create all pauli strings from OrderNObservables.

    Args:
        s (OrderNObservables): settings object.

    Returns:
        list[qt.Qobj]: list of all observables upto order n.
    """

    # read settings
    m = s.m
    n = s.n
    return order_n_observables(m=m, n=n)


def _all_observables(s: AllObservables) -> list[qt.Qobj]:
    """Convenience function to create all pauli strings from AllObservables.

    Args:
        s (AllObservables): settings object.

    Returns:
        list[qt.Qobj]: list of all observables.
    """

    # read settings
    m = s.m
    return all_observables(m=m)


def create_observables(s: Observables) -> list[qt.Qobj]:
    """Convenience function to create observables from (subclass of) Observables object.

    Args:
        s (Observables): settings object.

    Returns:
        list[qt.Qobj]: list of observables corresponding to the settings.
    """

    if isinstance(s, KRandomObservables):
        return _k_random_observables(s)
    if isinstance(s, OrderNObservables):
        return _order_n_observables(s)
    if isinstance(s, AllObservables):
        return _all_observables(s)
