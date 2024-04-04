"""
Dataclasses that represent possible sets of Observables
and some function to then create those lists of observables.

References"
    Original code by @lviss (my_function.py)

Info:
    Created on Thu Apr 4 2024

    @author: davidvrchen
"""

from typing import Iterable
from dataclasses import dataclass
import random as rd
from itertools import product, chain

import qutip as qt
from more_itertools import distinct_permutations

if __name__ == "__main__":
    from pauli_spin_matrices import SPIN_MATRIX_DICT, SPIN_MATRICES_LST
else:
    from .pauli_spin_matrices import SPIN_MATRIX_DICT, SPIN_MATRICES_LST


@dataclass
class Observables:
    """Dataclass that represents a set of observables on m qubits.

    Args:
    -----
    m (int): number of qubits.
    """

    m: int


@dataclass
class kRandomObservables(Observables):
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
class OrdernObservables(Observables):
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


def k_random_pauli_strs(m: int, k: int, seed: int = None):
    """Randomly select k Pauli strings from all possible pauli strings on m qubits.

    >>> k_random_pauli_strs(3, 5).__len__()
    5
    """

    if not seed is None:
        rd.seed(seed)

    all_pauli_strs = list(product(SPIN_MATRICES_LST, repeat=m))
    pauli_strs = rd.sample(all_pauli_strs, k)

    return pauli_strs


def order_n_pauli_strs(m: int, n: int):
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


def all_pauli_strs(m: int):
    """All Pauli strings on m qubits.

    >>> all_pauli_strs(1)
    [('I',), ('X',), ('Y',), ('Z',)]

    >>> len( all_pauli_strs(3) ) # 4**3
    64
    """
    pauli_strs = product(SPIN_MATRICES_LST, repeat=m)

    return list(pauli_strs)


# conversion of pauli string to operator
def pauli_str_2_op(pauli_str: Iterable[str]):
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


# convenience function to convert list of pauli strings to operators
def pauli_strs_2_ops(pauli_strs: list[Iterable[str]]):
    """Apply pauli_str_2_op on all pauli strings in pauli_strs."""
    return [pauli_str_2_op(pauli_str) for pauli_str in pauli_strs]


# some convenience functions
def k_random_observables(s: kRandomObservables):
    """Convenience function to create all pauli strings from kRandomObservables."""
    # read settings
    m = s.m
    k = s.k
    seed = s.seed

    pauli_strs = k_random_pauli_strs(m=m, k=k, seed=seed)
    return pauli_strs_2_ops(pauli_strs)


def order_n_observables(s: OrdernObservables):
    """Convenience function to create all pauli strings from OrdernObservables."""
    # read settings
    m = s.m
    n = s.n

    pauli_strs = order_n_pauli_strs(m=m, n=n)
    return pauli_strs_2_ops(pauli_strs)


def all_observables(s: AllObservables):
    """Convenience function to create all pauli strings from AllObservables."""
    # read settings
    m = s.m

    pauli_strs = all_pauli_strs(m=m)
    return pauli_strs_2_ops(pauli_strs)


def create_observables(s: Observables):
    """Convenience function to create observables from (subclass of) Observables object."""
    if isinstance(s, kRandomObservables):
        return k_random_observables(s)
    if isinstance(s, OrdernObservables):
        return order_n_observables(s)
    if isinstance(s, AllObservables):
        return all_observables(s)


if __name__ == "__main__":
    import doctest

    MY_FLAG = doctest.register_optionflag("ELLIPSIS")
    doctest.testmod(verbose=True, optionflags=MY_FLAG)
