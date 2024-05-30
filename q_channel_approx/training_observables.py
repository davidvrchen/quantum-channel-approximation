"""
Dataclasses that represent possible sets of Observables,
some function to then create those observables and some other .
"""

from typing import Iterable
from dataclasses import dataclass
from itertools import product, chain

import numpy as np
import qutip as qt
from more_itertools import distinct_permutations

from q_channel_approx.physics_defns import SPIN_MATRIX_DICT, SPIN_MATRICES_LST


def all_pauli_strs(m: int) -> list[str]:
    """All Pauli strings on m qubits.

    >>> all_pauli_strs(1)
    [('I',), ('X',), ('Y',), ('Z',)]

    >>> len( all_pauli_strs(3) ) # 4**3
    64
    """
    pauli_strs = product(SPIN_MATRICES_LST, repeat=m)

    return list(pauli_strs)


def pauli_str_2_op(pauli_str: Iterable[str]) -> qt.Qobj:
    """Create the Pauli string operator from a Pauli string.

    >>> pauli_str_2_op( ("I", "I") )
    Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
    Qobj data =
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    """

    pauli_str = [SPIN_MATRIX_DICT[pauli_mat] for pauli_mat in pauli_str]

    return qt.tensor(*pauli_str)


def pauli_strs_2_ops(pauli_strs: list[Iterable[str]]) -> list[qt.Qobj]:
    """Apply pauli_str_2_op on all pauli strings in pauli_strs."""
    return [pauli_str_2_op(pauli_str) for pauli_str in pauli_strs]


def k_random_pauli_strs(m: int, k: int, seed: int = None) -> list[str]:
    """Randomly select k Pauli strings from all possible pauli strings on m qubits.

    >>> k_random_pauli_strs(3, 5).__len__()
    5
    """

    if seed is None:
        rng = np.random.default_rng()
        seed = rng.integers(10**4)
        print(f"k_random_pauli_strs: {seed=}")

    rng = np.random.default_rng(seed=seed)

    pauli_strs = rng.choice(all_pauli_strs(m), size=k, replace=False)

    return pauli_strs


def order_n_pauli_strs(m: int, n: int) -> list[str]:
    """All pauli strings on m qubits upto and including order n.

    >>> order_n_pauli_strs(3, 0)
    [('I', 'I', 'I')]

    >>> order_n_pauli_strs(3, 1)
    [('I', 'I', 'I'), ('I', 'I', 'X'), ('I', 'I', 'Y'), ('I', 'I', 'Z'), ('I', 'X', 'I'), ('I', 'Y', 'I'), ('I', 'Z', 'I'), ('X', 'I', 'I'), ('Y', 'I', 'I'), ('Z', 'I', 'I')]
    """

    pauli_strs = []
    for k in range(n + 1):
        pauli_templates = ["XYZ" for i in range(k)] + ["I" for i in range(m - k)]
        permuted_templates = list(distinct_permutations(pauli_templates))

        # chain all Pauli strings of order k to the end of the list
        pauli_strs = chain.from_iterable(
            (
                pauli_strs,
                chain.from_iterable(
                    product(*template) for template in permuted_templates
                ),
            )
        )

    return list(pauli_strs)


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
    pauli_strs = k_random_pauli_strs(m=m, k=k, seed=seed)
    return pauli_strs_2_ops(pauli_strs)


def order_n_observables(m: int, n: int) -> list[qt.Qobj]:
    pauli_strs = order_n_pauli_strs(m=m, n=n)
    return pauli_strs_2_ops(pauli_strs)


def all_observables(m: int) -> list[qt.Qobj]:
    pauli_strs = all_pauli_strs(m=m)
    return pauli_strs_2_ops(pauli_strs)


def _k_random_observables(s: KRandomObservables) -> list[qt.Qobj]:
    """Convenience function to create all pauli strings from kRandomObservables."""
    # read settings
    m = s.m
    k = s.k
    seed = s.seed

    return k_random_observables(m=m, k=k, seed=seed)


def _order_n_observables(s: OrderNObservables) -> list[qt.Qobj]:
    """Convenience function to create all pauli strings from OrdernObservables."""
    # read settings
    m = s.m
    n = s.n
    return order_n_observables(m=m, n=n)


def _all_observables(s: AllObservables) -> list[qt.Qobj]:
    """Convenience function to create all pauli strings from AllObservables."""
    # read settings
    m = s.m
    return all_observables(m=m)


def create_observables(s: Observables) -> list[qt.Qobj]:
    """Convenience function to create observables from (subclass of) Observables object."""
    if isinstance(s, KRandomObservables):
        return _k_random_observables(s)
    if isinstance(s, OrderNObservables):
        return _order_n_observables(s)
    if isinstance(s, AllObservables):
        return _all_observables(s)


if __name__ == "__main__":
    import doctest

    MY_FLAG = doctest.register_optionflag("ELLIPSIS")
    doctest.testmod(verbose=True, optionflags=MY_FLAG)
