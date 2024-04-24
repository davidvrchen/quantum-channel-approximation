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

from numba.experimental import jitclass

import numpy as np
import qutip as qt
from q_lab_toolbox.training_data import measure_rhos
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

    def _U(self, theta):
        """Return U but as qt.Qobj"""

        return qt.Qobj(self.U(theta), dims=[[2] * self.n_qubits, [2] * self.n_qubits])

    def J(self, flat_theta, training_data) -> float:
        """Calculate loss function J."""

        # reshape theta such that U understands it, need to make phi'
        theta = self.reshape_theta(flat_theta)

        # read initial state, list of observables and matrix of expectations
        Os, rho0s, Ess = training_data

        # recall dimensions of expectation matrix
        # we need to match these dimensions when constructing the
        # expectations of phi'
        L, K, _N = Ess.shape

        rhohatss = np.zeros((L, _N, 2**self.m, 2**self.m), dtype=np.complex128)

        for l, rho0 in enumerate(rho0s):
            rhohatss[l, :, :, :] = self.approximate_evolution(theta, rho0, _N - 1)

        Ehatss = np.zeros((L, K, _N), dtype=np.float64)

        for l, rhohats in enumerate(rhohatss):
            Ehatss[l, :, :] = measure_rhos(rhohats, Os)

        tracess = Ehatss - Ess

        tracess = tracess * tracess

        error = np.sum(tracess)
        return error

    def approximate_evolution(self, theta, rho0, N):
        """Use theta to approximate the evolution of the
        state rho for N steps.

        Args:
        -----
            theta: parameters to use

            rho0: initial state

            N: number timesteps to approximate

        Returns:
        --------
            rhos: list of the approximated states
            note this list is of length N+1 as it
            includes rho0
        """
        # evolution dicatates by parameters
        phi_prime = self.phi_prime(theta)

        # calculate the rhos
        rho_acc = rho0
        rhos = [rho0]
        for _ in range(N):
            rho_acc = phi_prime(rho_acc)
            rhos.append(rho_acc)

        return rhos

    def phi_prime(self, theta):
        """Returns phi prime, the quantum channel."""

        U = self._U(theta)

        def _phi_prime(rho):
            """Tr_b [ U[theta] (rho x ancilla) U[theta]^dag ]"""
            full_system = qt.tensor(
                qt.Qobj(rho, dims=[[2] * self.m, [2] * self.m]), self.ancilla
            )

            return qt.ptrace(U * full_system * U.dag(), range(self.m))

        return _phi_prime

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

        return list(chain.from_iterable(reshape(self.theta_shapes, flat_theta)))

    def flatten_theta(self, theta):
        """Flatten theta (as U understands it) for the optimization step.
        Note: reshape_theta and flatten_theta must be inverses."""
        all_pars = [np.ravel(par) for par in theta]
        return np.concatenate(all_pars)


class HardwareEfficientAnsatz(GateBasedUnitaryCircuit):

    def __init__(
        self,
        *,
        depth,
        m,
        n_qubits,
        gate_type,
        structure,
    ):
        self.depth = depth
        super().__init__(
            m=m,
            n_qubits=n_qubits,
            structure=structure,
            gate_type=gate_type,
        )

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

        qubit_pars = np.ones([self.depth, self.n_qubits, 3]) * (np.pi / 4)

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

        return qc


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
