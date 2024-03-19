"""
Provides a classes that define various parametrized unitary circuits.


References:
    original code for decay_examply by @lviss

Info:
    Created on Tue March 19 2024

    @author: davidvrchen
"""

from abc import ABC, abstractmethod

import numpy as np
import qutip as qt
import scipy as sc

from q_lab_toolbox.utils.my_functions import generate_gate_connections


class GateBasedUnitaryCircuit(ABC):

    def __init__(
        self,
        m,
        circuit_type="ryd",
        structure="triangle",
        depth=0,
        **kwargs,
    ):
        self.m = m
        self.depth = depth

        self.gate_type = circuit_type
        self.pairs = generate_gate_connections(m, structure=structure)

        if circuit_type == "ryd":
            self.init_ryd(**kwargs)
        elif circuit_type in ["cnot", "xy", "xy_var", "decay"]:
            self.init_gates(circuit_type)
        else:
            print("Coupling gate type {} unknown".format(circuit_type))

    def init_ryd(self, **kwargs):

        def gate(gate_par):
            return rydberg_pairs(self.m, self.pairs, t_ryd=gate_par)

        self.entangle_gate = gate

    def init_gates(self, circuit_type):
        gate_dict = {"cnot": cnot_gate_ij, "xy": gate_xy, "decay": gate_decay}

        def gate(gate_par):
            return circuit_pairs(self.m, self.pairs, gate_dict[circuit_type], gate_par)

        self.entangle_gate = gate

    def _rz_gate(self, theta_list):
        gate = np.array([[1]])
        for theta in theta_list:
            gate_single = np.array(
                [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]]
            )
            gate = np.kron(gate, gate_single)
        return gate

    def _rx_gate(self, theta_list):
        gate = np.array([[1]])
        for theta in theta_list:
            gate_single = np.array(
                [
                    [np.cos(theta / 2), -np.sin(theta / 2)],
                    [np.sin(theta / 2), np.cos(theta / 2)],
                ]
            )
            gate = np.kron(gate, gate_single)
        return gate


    @abstractmethod
    def U(self, theta):
        """Returns qt.Qobj operator that represents the unitary circuits with parameters theta."""

    @abstractmethod
    def init_theta(self):
        """Creates array / data object that the parametrized circuit U understands."""


class HardwareAnsatz(GateBasedUnitaryCircuit):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_theta(self):
        """U needs a triple
            qubit_pars (was theta),
            ent_pars (was gate_par)
            n 
        to return numerical values for the unitary as a matrix."""

        qubit_pars = np.ones([self.depth, self.m + 1, 3]) * (np.pi/2)
        pars_per_layer = len(generate_gate_connections(2*self.m+1, structure=self.qubit_structure, cutoff=True))
        ent_pars = np.ones([self.depth, pars_per_layer]) * self.phi
        return qubit_pars, ent_pars, n



    def U(self, theta):
        """Returns the numerical value of the hardware ansatz unitary.
        (parametrized by theta)"""

        qubit_pars, ent_pars, n = theta

        depth, m = qubit_pars[:, :, 0].shape
        
        # start creating the quantum circuit
        qc = np.identity(2**m)

        for k in range(depth):

            # z-x-z gates with parameters theta
            qc = (
                qc
                @ self._rz_gate(theta[k, :, 0])
                @ self._rx_gate(theta[k, :, 1])
                @ self._rz_gate(theta[k, :, 2])
            )
            try:
                self.entangle_gate(gate_par=ent_pars[k, :])
            except ValueError:
                print("Problem in circuit gate generation")
                print(ent_pars)
                print(ent_pars[k, :])
                raise

            qc = qc @ self.entangle_gate(gate_par=ent_pars[k, :])

        qc = qc @ np.linalg.matrix_power(qc, n)

        return qc


class HardwareAnsatzWithH(GateBasedUnitaryCircuit):


    def init_theta(self):
        """same parameters as hardware ansatz but extra t_ham"""

    def U(self, theta, gate_par, n, t):

        depth, m = theta[:, :, 0].shape

        # to parametrize the entanglement
        gate_par = self.set_gate_par(depth, gate_par)

        # start creating the quantum circuit
        qc = np.identity(2**m)

        for k in range(depth):
            # print(self.H)

            # hamiltonian for t_ham

            H = sc.linalg.expm(-(1j) * self.t_ham * self.H)
            H = qt.Qobj(H)
            H.dims = [[2] * 2, [2] * 2]
            H_q = qt.expand_operator(H, m, range(m))
            qc = qc @ H_q.full()

            # z-x-z gates with parameters theta
            qc = (
                qc
                @ self._rz_gate(theta[k, :, 0])
                @ self._rx_gate(theta[k, :, 1])
                @ self._rz_gate(theta[k, :, 2])
            )
            try:
                self.entangle_gate(gate_par=gate_par[k, :])
            except ValueError:
                print("Problem in circuit gate generation")
                print(gate_par)
                print(gate_par[k, :])
                raise
            qc = qc @ self.entangle_gate(gate_par=gate_par[k, :])

        qc = qc @ np.linalg.matrix_power(qc, n)

        return qc



def circuit_pairs(m, pairs, gate_fun, gate_par):

    gate = np.eye(2**m)

    for i, (k, l, d) in enumerate(pairs):
        gate = (
            gate
            @ qt.qip.operations.gates.expand_operator(
                gate_fun(gate_par[i]), m, [k, l]
            ).full()
        )

    return gate


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

