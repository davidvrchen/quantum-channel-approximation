"""
Provides a classes that define various parametrized unitary circuits.


References:
    original code for decay_examply by @lviss

Info:
    Created on Tue March 19 2024

    Last update on Mon Apr 8 2024

    @author: davidvrchen
"""

from operator import mul
from itertools import chain
from functools import reduce
from abc import ABC, abstractmethod


import numpy as np
import qutip as qt
import scipy as sc

from q_lab_toolbox.my_functions import generate_gate_connections


class GateBasedUnitaryCircuit(ABC):

    def __init__(
        self,
        m,
        n_qubits,
        gate_type,
        structure,
    ):

        self.m = m
        self.n_qubits = n_qubits
        n_ancillas = n_qubits - m

        self.structure = structure

        self.gate_type = gate_type
        self.pairs = generate_gate_connections(n_qubits, structure=structure)

        if gate_type == "ryd":
            self.init_ryd()
        elif gate_type in ["cnot", "xy", "xy_var", "decay"]:
            self.init_gates(gate_type)
        else:
            print("Coupling gate type {} unknown".format(gate_type))

        # pre-compute some things that will be needed frequently in other functions
        self.theta_shapes = self.get_theta_shapes()

        state00 = qt.Qobj([[1, 0], [0, 0]])
        self.ancilla = qt.tensor([state00 for _ in range(n_ancillas)])

    def init_ryd(self):

        def gate(gate_par):
            return rydberg_pairs(self.n_qubits, self.pairs, t_ryd=gate_par)

        self.entangle_gate = gate




def rydberg_pairs(m, pairs, t_ryd):
    rydberg = np.zeros([4, 4])
    rydberg[3, 3] = 1
    rydberg_2gate = qt.Qobj(rydberg, dims=[[2] * 2, [2] * 2])

    rydberg_gate = np.zeros([2**m, 2**m], dtype=np.complex128)
    for k, l, d in pairs:
        ham = qt.qip.operations.gates.expand_operator(rydberg_2gate, m, [k, l]).full()
        rydberg_gate += ham / d**3  # distance to the power -6

    return sc.linalg.expm(-1j * t_ryd[0] * rydberg_gate)


def gate_xy(phi):
    if type(phi) != np.float64:
        print("parameter error")
        print(phi)
        print(type(phi))
        raise ValueError
    gate_xy = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi / 2), -1j * np.sin(phi / 2), 0],
            [0, -1j * np.sin(phi / 2), np.cos(phi / 2), 0],
            [0, 0, 0, 1],
        ]
    )
    return qt.Qobj(gate_xy, dims=[[2] * 2, [2] * 2])


def gate_decay(gammat):
    # print(gammat)
    if gammat < 0:
        gammat = 0
    gate_decay = np.array(
        [
            [1, 0, 0, 0],
            [0, -np.exp(-gammat / 2), (1 - np.exp(-gammat)) ** (1 / 2), 0],
            [0, (1 - np.exp(-gammat)) ** (1 / 2), np.exp(-gammat / 2), 0],
            [0, 0, 0, 1],
        ]
    )
    return qt.Qobj(gate_decay, dims=[[2] * 2, [2] * 2])


def cnot_gate_ij(offset):
    if offset == 0:
        gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    else:
        gate = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    gate = qt.Qobj(gate, dims=[[2] * 2, [2] * 2])
    return gate
