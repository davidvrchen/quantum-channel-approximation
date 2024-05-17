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


def _k_random_observables(m: int, k: int, seed: int):
    pauli_strs = k_random_pauli_strs(m=m, k=k, seed=seed)
    return pauli_strs_2_ops(pauli_strs)


# some convenience functions
def k_random_observables(s: kRandomObservables):
    """Convenience function to create all pauli strings from kRandomObservables."""
    # read settings
    m = s.m
    k = s.k
    seed = s.seed

    return _k_random_observables(m=m, k=k, seed=seed)


def _order_n_observables(m: int, n: int):
    pauli_strs = order_n_pauli_strs(m=m, n=n)
    return pauli_strs_2_ops(pauli_strs)


def order_n_observables(s: OrdernObservables):
    """Convenience function to create all pauli strings from OrdernObservables."""
    # read settings
    m = s.m
    n = s.n
    return _order_n_observables(m=m, n=n)


def _all_observables(m: int):
    pauli_strs = all_pauli_strs(m=m)
    return pauli_strs_2_ops(pauli_strs)


def all_observables(s: AllObservables):
    """Convenience function to create all pauli strings from AllObservables."""
    # read settings
    m = s.m
    return _all_observables(m=m)


import itertools

import qutip as qt

from .target_systems import TargetSystem


if __name__ == "__main__":
    import os, sys

    module_dir = os.getcwd()
    import_file = f"{module_dir}/my_combinators.py"
    print(import_file, os.getcwd())
    sys.path.append(os.path.dirname(os.path.expanduser(import_file)))



def read_00_op(tup) -> qt.Qobj:
    """Read state \|0><0\| on ith qubit in m qubit basis

    Args:
        tup (i, m): ith qubit, m qubit basis

    Returns:
        qt.Qobj: readout operator
    """
    i, m = tup
    op_00 = qt.Qobj([[1, 0], [0, 0]])
    return qt.qip.expand_operator(op_00, m, (i,))


def read_11_op(tup) -> qt.Qobj:
    """Read state \|1><1\| on ith qubit in m qubit basis

    Args:
        tup (i, m): ith qubit, m qubit basis

    Returns:
        qt.Qobj: the readout operator
    """

    i, m = tup
    op_11 = qt.Qobj([[0, 0], [0, 1]])
    return qt.qip.expand_operator(op_11, m, (i,))


read_op_pair = lambda x : (read_00_op(x), read_11_op(x))


def create_readout_individual_qs(s: TargetSystem) -> list[qt.Qobj]:
    """Create list of readout operators for each qubit in
    m qubit basis.
    Each qubit is read out as \|0><0\| and \|1><1\|

    Args:
        s (TargetSystemSettings): settings

    >>> create_readout_ops( TargetSystemSettings(m=1, gammas=(0,) ))[0]
    Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
    Qobj data =
    [[1. 0.]
     [0. 0.]]

    >>> create_readout_ops( TargetSystemSettings(m=1, gammas=(0,) ))[1]
    Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
    Qobj data =
    [[0. 0.]
     [0. 1.]]
    """

    # read parameters from settings
    m = s.m

    ops = [read_op_pair((i, m)) for i in range(m)]
    return list(itertools.chain.from_iterable(ops))


def b2op(b):
    if b == "0":
        return qt.Qobj([[1, 0], [0, 0]])
    if b == "1":
        return qt.Qobj([[0, 0], [0, 1]])


def str2op(bs):
    return [b2op(b) for b in bs]


def str2tensor(bs):
    return qt.tensor(str2op(bs))

def computation_basis_labels(s: TargetSystem):
    m = s.m
    
    labels = [rf"$|{format(i, f'0{m}b')}\rangle \langle{format(i, f'0{m}b')}|$" for i in range(2**m)]

    return labels

def create_readout_computational_basis(s: TargetSystem):
    """to be added"""

    # read parameters from settings
    m = s.m

    comp_basis = range(2**m)

    ops = [str2tensor(format(bs, f"0{m}b")) for bs in comp_basis]
    return ops


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
