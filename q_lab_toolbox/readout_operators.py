"""
to be documented

References"
    Original code by @lviss (my_function.py)

Info:
    Created on Thu Apr 4 2024

    @author: davidvrchen
"""

from dataclasses import dataclass
import math
import random as rd
from itertools import product


import numpy as np
import qutip as qt
from more_itertools import distinct_permutations


if __name__ == "__main__":
    from .pauli_spin_matrices import SPIN_MATRIX_DICT, SPIN_MATRICES_LST
else:
    from pauli_spin_matrices import SPIN_MATRIX_DICT, SPIN_MATRICES_LST


@dataclass
class Observables:
    m: int
    space: str


@dataclass
class kRandomObservables(Observables):
    k: int


@dataclass
class OrdernObservables(Observables):
    n: int


@dataclass
class AllObservables(Observables):


def _k_random_observables(m: int, k: int):
    
    pauli_list_names_full = list(product(SPIN_MATRICES_LST, repeat = m))
    pauli_list_names = rd.sample(pauli_list_names_full, k)

    return k, pauli_list_names


def _order_n_observables(m: int, n: int):

    num_pauli = sum(3**i * math.comb(m,i) for i in range(n+1))

    pauli_list_names = []
    for k in range(n+1):
        terms_pauli = ['XYZ' for i in range(k)] + ['I' for i in range(m-k)]
        terms_permuted = list(distinct_permutations(terms_pauli))
        for name_list in terms_permuted:
            pauli_list_names = pauli_list_names + list(product(*name_list))

    return num_pauli, pauli_list_names


def _all_observables(m: int):
    num_pauli = 4**m
    pauli_list_names = list(product(SPIN_MATRICES_LST, repeat = m))


def k_random_observables(s: kRandomObservables):
    # read settings from kRandomObservables
    m = s.m
    k = s.k

    return _k_random_observables(m=m, k=k)


def order_n_observables(s: OrdernObservables):
    # read settings from OrdernObservables
    m = s.m
    n = s.n

    return _order_n_observables(m=m, n=n)


def all_observables(s: AllObservables):
    # read settings from AllObservables
    m = s.m
    
    return _all_observables(m=m)


def pauli_from_str(pauli_str: str):
    """Create the Pauli string operator from a Pauli string."""

    pauli_str = [SPIN_MATRIX_DICT[pauli_mat] for pauli_mat in pauli_str]

    return qt.tensor(*pauli_str)


def create_observables(s: Observables):
    """Conveniece function to create the appropriate observables from
    Observables object.
    """
    
    if isinstance(s, kRandomObservables):
        num_pauli, pauli_list_names = k_random_observables(s)
    if isinstance(s, OrdernObservables):
        num_pauli, pauli_list_names = order_n_observables(s)
    if isinstance(s, kRandomObservables):
        num_pauli, pauli_list_names = all_observables(s)
    
    pauli_list_names = np.array(pauli_list_names)
    
    pauli_list = np.ndarray([num_pauli, 2**m, 2**m], dtype = np.complex_)
    id_qubit_list = np.zeros([m, num_pauli])
    for i, name in enumerate(pauli_list_names):
        pauli_list[i,:,:] = pauli_from_str(name)
        for j in range(m):
            if name[j] != "I":
                id_qubit_list[j,i]=1


    n, pauli_index_x, pauli_index_y = np.where(pauli_list != 0)
    pauli_index_factor = pauli_list[n, pauli_index_x, pauli_index_y]
    
    pauli_index_x = pauli_index_x.reshape(num_pauli,2**m)
    pauli_index_y = pauli_index_y.reshape(num_pauli,2**m)
    pauli_index_factor = pauli_index_factor.reshape(num_pauli,2**m)
    index_list = (pauli_index_x, pauli_index_y, pauli_index_factor)
    
    
    return pauli_list, pauli_list_names, id_qubit_list, index_list



if __name__ == "__main__":
    import doctest

    MY_FLAG = doctest.register_optionflag("ELLIPSIS")
    doctest.testmod(verbose=True, optionflags=MY_FLAG)
