"""
Functions related to the creation of Pauli strings.
"""

from typing import Iterable
from itertools import product, chain

import numpy as np
import qutip as qt
from more_itertools import distinct_permutations

from q_channel_approx.physics_defns import SPIN_MATRIX_DICT, SPIN_MATRICES_LST


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


def all_pauli_strs(m: int) -> list[str]:
    """All Pauli strings on m qubits.

    >>> all_pauli_strs(1)
    [('I',), ('X',), ('Y',), ('Z',)]

    >>> len( all_pauli_strs(3) ) # 4**3
    64
    """
    pauli_strs = product(SPIN_MATRICES_LST, repeat=m)

    return list(pauli_strs)


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
