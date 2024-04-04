from dataclasses import dataclass
import math
import random as rd
import re
from itertools import product

import qutip as qt

from more_itertools import distinct_permutations

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
    
    pauli_list_names_full = list(product(PAULI_MATRICES, repeat = m))
    pauli_list_names = rd.sample(pauli_list_names_full, k)

    return k, pauli_list_names, pauli_list_names_full

def _order_n_observables(m: int, n: int):

    num_pauli = sum(3**i * math.comb(m,i) for i in range(n+1))

    pauli_list_names = []
    for k in range(n+1):
        terms_pauli = ['XYZ' for i in range(k)] + ['I' for i in range(m-k)]
        terms_permuted = list(distinct_permutations(terms_pauli))
        for name_list in terms_permuted:
            pauli_list_names = pauli_list_names + list(product(*name_list))

    return num_pauli, pauli_list_names, 


def _all_observables(m: int):
    pass




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

def create_observables(s: Observables):
    """Conveniece function to create the appropriate observables from
    Observables object.
    """
    
    if isinstance(s, kRandomObservables):
        return k_random_observables(s)
    if isinstance(s, OrdernObservables):
        return order_n_observables(s)
    if isinstance(s, kRandomObservables):
        return all_observables(s)
    
    pauli_list_names = np.array(pauli_list_names)
    
    pauli_list = np.ndarray([num_pauli, 2**m, 2**m], dtype = np.complex_)
    id_qubit_list = np.zeros([m, num_pauli])
    for i, name in enumerate(pauli_list_names):
        pauli_list[i,:,:] = pauli_from_str(name)
        for j in range(m):
            if name[j] != "I":
                id_qubit_list[j,i]=1
                
    #pauli_index_x = np.array(2**m)
    #pauli_index_y = np.array(2**m)
    #pauli_index_factor = np.array(2**m)
    n, pauli_index_x, pauli_index_y = np.where(pauli_list != 0)
    pauli_index_factor = pauli_list[n, pauli_index_x, pauli_index_y]
    
    pauli_index_x = pauli_index_x.reshape(num_pauli,2**m)
    pauli_index_y = pauli_index_y.reshape(num_pauli,2**m)
    pauli_index_factor = pauli_index_factor.reshape(num_pauli,2**m)
    index_list = (pauli_index_x, pauli_index_y, pauli_index_factor)
    
    
    return pauli_list, pauli_list_names, id_qubit_list, index_list