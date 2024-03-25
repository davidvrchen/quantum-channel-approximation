"""
Provides a classes that define various parametrized unitary circuits.


References:
    original code for decay_examply by @lviss

Info:
    Created on Tue March 19 2024

    @author: davidvrchen
"""

from operator import mul
from itertools import chain
from functools import reduce
from abc import ABC, abstractmethod

import numpy as np
import qutip as qt
import scipy as sc

from q_lab_toolbox.utils.my_functions import generate_gate_connections


class GateBasedUnitaryCircuit(ABC):

    def __init__(
        self,
        n_qubits,
        gate_type="ryd",
        structure="triangle",
        depth=0,
        **kwargs,
    ):
        self.n_qubits = n_qubits
        self.depth = depth
        self.structure = structure

        self.gate_type = gate_type
        self.pairs = generate_gate_connections(n_qubits, structure=structure)

        if gate_type == "ryd":
            self.init_ryd(**kwargs)
        elif gate_type in ["cnot", "xy", "xy_var", "decay"]:
            self.init_gates(gate_type)
        else:
            print("Coupling gate type {} unknown".format(gate_type))

        self.theta_shapes = self.get_theta_shapes()

    def init_ryd(self, **kwargs):

        def gate(gate_par):
            return rydberg_pairs(self.n_qubits, self.pairs, t_ryd=gate_par)

        self.entangle_gate = gate

    def init_gates(self, circuit_type):
        gate_dict = {"cnot": cnot_gate_ij, "xy": gate_xy, "decay": gate_decay}

        def gate(gate_par):
            return circuit_pairs(
                self.n_qubits, self.pairs, gate_dict[circuit_type], gate_par
            )

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

    def init_flat_theta(self):
        """Convenience method that creates theta as U understands and immediately flattens it."""
        return self.flatten_theta(self.init_theta())

    def get_theta_shapes(self):
        """Returns a list of parameter arrays shapes. The shapes that U understands for theta."""
        shapes = [s.shape for s in self.init_theta()]
        return shapes

    def reshape_theta(self, flat_theta):
        """Turn flat theta back into array that U understands.
        Note: reshape_theta and flatten_theta must be inverses."""

        def reshape(ss, xs):
            """reshape xs according to tuples of shapes defined is ss.
            Note ss is a list of tuples and xs is 1d numpy array."""
            if not ss and not xs.size:
                return []

            index = reduce(mul, ss[0])
            head, tail = xs[:index], xs[index:]

            reshaped_head = np.reshape(head, ss[0])
            reshaped_tail = list(chain.from_iterable(reshape(ss[1:], tail)))

            # reshaped_head needs to be packaged up as a list
            # because chain flattens one level
            return chain([[reshaped_head], reshaped_tail])

        return list(chain.from_iterable(reshape(self.theta_shapes, flat_theta) ))

    def flatten_theta(self, theta):
        """Flatten theta (as U understands it) for the optimization step.
        Note: reshape_theta and flatten_theta must be inverses."""
        all_pars = [np.ravel(par) for par in theta]
        return np.concatenate(all_pars)


class HardwareAnsatz(GateBasedUnitaryCircuit):

    def __init__(
        self,
        *,
        depth,
        n_qubits,
        n_repeats=1,
        circuit_type="ryd",
        structure="triangle",
        **kwargs,
    ):
        super().__init__(
            n_qubits=n_qubits,
            circuit_type=circuit_type,
            structure=structure,
            depth=depth,
            **kwargs,
        )
        self.n_repeats = n_repeats

    def shape_ent_pars(self, depth, gate_par):
        """documentation to be added..."""
        if isinstance(gate_par, float) or isinstance(gate_par, int):
            gate_par = np.zeros([depth, len(self.pairs)]) + gate_par

        elif gate_par.shape == (depth, 1):
            gate_par = np.array([[par] * len(self.pairs) for par in gate_par])

        elif gate_par.shape == (depth, len(self.pairs)):
            pass

        else:
            print("Gate parameters invalid dimensions of", gate_par.shape)
            print(depth, len(self.pairs))
            raise IndexError

        if self.gate_type == "cnot":
            gate_par = np.zeros([depth, len(self.pairs)])
            for i in range((depth) // 2):
                gate_par[2 * i + 1, :] = 1

        return gate_par

    def init_theta(self):
        """U needs a pair
            qubit_pars (was theta),
            ent_pars (was gate_par)
        to return numerical values for the unitary as a matrix."""

        qubit_pars = np.ones([self.depth, self.n_qubits, 3]) * (np.pi / 2)

        ent_pars = self.shape_ent_pars(self.depth, 1)
        return qubit_pars, ent_pars

    def U(self, theta):
        """Returns the numerical value of the hardware ansatz unitary.
        (parametrized by theta)"""

        qubit_pars, ent_pars = theta

        depth, m = qubit_pars[:,:,0].shape

        # start creating the quantum circuit
        qc = np.identity(2**m)

        for k in range(depth):

            # z-x-z gates with parameters qubit_pars
            qc = (
                qc
                @ self._rz_gate(qubit_pars[k, :, 0])
                @ self._rx_gate(qubit_pars[k, :, 1])
                @ self._rz_gate(qubit_pars[k, :, 2])
            )
            try:
                self.entangle_gate(gate_par=ent_pars[k, :])
            except ValueError:
                print("Problem in circuit gate generation")
                print(ent_pars)
                print(ent_pars[k, :])
                raise

            # entanglement with ent_pars
            qc = qc @ self.entangle_gate(gate_par=ent_pars[k, :])

        # repeat the
        qc = qc @ np.linalg.matrix_power(qc, self.n_repeats)

        return qc


class HardwareAnsatzSepH(GateBasedUnitaryCircuit):

    def shape_ent_pars(self, depth, gate_par):
        """documentation to be added..."""
        if isinstance(gate_par, float) or isinstance(gate_par, int):
            gate_par = np.zeros([depth, len(self.pairs)]) + gate_par

        elif gate_par.shape == (depth, 1):
            gate_par = np.array([[par] * len(self.pairs) for par in gate_par])

        elif gate_par.shape == (depth, len(self.pairs)):
            pass

        else:
            print("Gate parameters invalid dimensions of", gate_par.shape)
            print(depth, len(self.pairs))
            raise IndexError

        if self.gate_type == "cnot":
            gate_par = np.zeros([depth, len(self.pairs)])
            for i in range((depth) // 2):
                gate_par[2 * i + 1, :] = 1

        return gate_par

    def init_theta(self):
        """U needs a pair
            qubit_pars (was theta),
            ent_pars (was gate_par)
        to return numerical values for the unitary as a matrix."""

        qubit_pars = np.ones([self.depth, self.n_qubits, 3]) * (np.pi / 2)

        ent_pars = self.shape_ent_pars(self.depth, 1)
        return qubit_pars, ent_pars

    def U(self, theta):
        """Returns the numerical value of the hardware ansatz unitary.
        (parametrized by theta)"""

        qubit_pars, ent_pars = theta

        depth, m = qubit_pars[:, :, 0].shape

        # start creating the quantum circuit
        qc = np.identity(2**m)

        # hamiltonian for t_ham
        H = sc.linalg.expm(-(1j) * self.t_ham * self.H)
        H = qt.Qobj(H)
        H.dims = [[2] * 2, [2] * 2]
        H_q = qt.expand_operator(H, m, range(m))
        qc = qc @ H_q.full()

        for k in range(depth):

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


class HardwareAnsatzWithH(GateBasedUnitaryCircuit):

    def shape_ent_pars(self, depth, gate_par):
        """documentation to be added..."""
        if isinstance(gate_par, float) or isinstance(gate_par, int):
            gate_par = np.zeros([depth, len(self.pairs)]) + gate_par

        elif gate_par.shape == (depth, 1):
            gate_par = np.array([[par] * len(self.pairs) for par in gate_par])

        elif gate_par.shape == (depth, len(self.pairs)):
            pass

        else:
            print("Gate parameters invalid dimensions of", gate_par.shape)
            print(depth, len(self.pairs))
            raise IndexError

        if self.gate_type == "cnot":
            gate_par = np.zeros([depth, len(self.pairs)])
            for i in range((depth) // 2):
                gate_par[2 * i + 1, :] = 1

        return gate_par

    def init_theta(self):
        """U needs a pair
            qubit_pars (was theta),
            ent_pars (was gate_par)
        to return numerical values for the unitary as a matrix."""

        qubit_pars = np.ones([self.depth, self.n_qubits, 3]) * (np.pi / 2)

        ent_pars = self.shape_ent_pars(self.depth, 1)
        return qubit_pars, ent_pars

    def U(self, theta):
        """Returns the numerical value of the hardware ansatz unitary.
        (parametrized by theta)"""

        qubit_pars, ent_pars = theta

        depth, m = qubit_pars[:, :, 0].shape

        # start creating the quantum circuit
        qc = np.identity(2**m)

        for k in range(depth):

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