# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:32:06 2023

@author: lviss
"""
import re
import time

import numpy as np
import qutip as qt
# import torch as to
import scipy as sc
import qutip as qt

from my_functions import generate_gate_connections


class U_circuit:

    def __init__(
        self, m, circuit_type="ryd", structure="triangle", t_ham=0, H=0, **kwargs
    ):
        """
        Class to generate a unitary circuit.

        To do: make faster (eg by saving calculated matrices)

        Parameters
        ----------
        m : int
            number of qubits (system A + supporting B (with one qubit more that system A)).
        circuit_type : string
            Gate type used to entangle the system qubits A with the supporting qubits B
        t_ham : float
            Interaction time for the hamiltonian between gates
        H : np.ndarray, 2**(m//2) x 2**(m//2)
            Hamiltonian acting on the system qubits A
        **kwargs : np.ndarray or float
            Initial parameters for fixed entanglement gates
            phi
            gammat
            t_ryd


        Returns
        -------
        None.

        """
        self.m = m
        self.t_ham = t_ham
        self.H = H

        # self.device = to.device('cuda' if to.cuda.is_available() else 'cpu')
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
    

    def set_gate_par(self, depth, gate_par):
        """documentation to be added...
        """
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

    def gate_circuit(
        self, theta, gate_par=1.0, n=1
    ):  # phi=0.0, n=1, gammat = 1.0, t_ryd = 0.1):
        """Create parametrized gate based unitary circuit.
        By optimizing the parameters of such a circuit,
        a unitary operator can be contructed that matches
        the system that is being modeled.

        Parameters:
        -----------
        theta: array of gate parameters

        gate_par: parameter for the entanglement gate
        """
        # print("making circuit; this is what theta looks like:")
        # print(theta)
        # print(f"this is the gate parameter: \n{gate_par}")

        depth, m = theta[:, :, 0].shape

        assert m == self.m, f"U_circuit class: Theta range incorrect. m={m} given, circuit defined on m={self.m}"

        # to parametrize the entanglement
        gate_par = self.set_gate_par(depth, gate_par)

        # start creating the quantum circuit
        qc = np.identity(2**m)


        for k in range(depth):
            # print(self.H)

            # hamiltonian for t_ham
            H_q = qt.expand_operator( qt.Qobj( sc.linalg.expm( -(1j) * self.t_ham * self.H) ) , m, (1,) )
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

            # entanglement for gate_par
            qc = qc @ self.entangle_gate(gate_par=gate_par[k, :])

        qc = qc @ np.linalg.matrix_power(qc, n)

        # import logging
        # logging.basicConfig(filename='example.log', encoding='utf-8')
        # logging.warning(qc)

        # qc = (
        #     qc
        #     @ self._rz_gate(theta[-1, :, 0])
        #     @ self._rx_gate(theta[-1, :, 1])
        #     @ self._rz_gate(theta[-1, :, 2])
        # )

        return qc


class Function:
    def __init__(self, qubit, sign, endtime, values):
        """
        Initialize the function class

        Parameters
        ----------
        qubit : int
            gives the control number.
        sign : -1,1
            sign of the function value.
        endtime : float
            total evolution time.
        values : np..ndarray, Zdt
            values for the new pulse

        Returns
        -------
        None.

        """
        self.qubit = qubit
        self.sign = sign
        self.values = values
        self.T = endtime

    def update(self, values):
        """
        Update the function parameters

        Parameters
        ----------
        values : np..ndarray, Zdt
            values for the new pulse

        Returns
        -------
        None.

        """
        self.values = values

    def f_t(self, t, args):
        """
        Returns the complex function value at the specified time

        Parameters
        ----------
        t : float
            specified time
        args : np.ndarray, Zdt x2
            functions values

        Returns
        -------
        float complex
            complex function value

        """
        index = int((t % self.T) // (self.T / len(self.values)))
        if index > (len(self.values) - 1):
            index = index - 1
        return self.values[index, 0] + self.sign * 1j * self.values[index, 1]


class U_circuit_pulse:

    def __init__(self, m, T, control_H, driving_H):
        self.m = m
        self.T = T
        self.control_H = control_H
        self.driving_H = driving_H
        self.n_controls = control_H.shape[0]

    def propagator(self, argsc, T, control_H, driving_H):
        """
        Determines the propagator as in the Fréchet derivatives

        Parameters
        ----------
        argsc : np.ndarray complex, num_control x Zdt x 2
            Describes the (complex) pulse parameters of the system throughout the process
        T : float
            Total evolution time.
        control_H : np.ndarray Qobj 2**m x 2**m, num_control x 2
            array of Qobj describing the control operators
        driving_H : Qobj 2**m x 2**m
            Hamiltonian describing the drift of the system

        Returns
        -------
        U : np.ndarray Qobjs, Zdt
             Array describing unitary evolution at each timestep
        """
        m, Zdt = argsc[:, :, 0].shape
        options = qt.Options()
        functions = np.ndarray(
            [
                m,
                2,
            ],
            dtype=object,
        )
        for k in range(m):
            functions[k, 0] = Function(k + 1, 1, T, argsc[k, :, :])
            functions[k, 1] = Function(k + 1, -1, T, argsc[k, :, :])
        H = [driving_H]
        for k in range(m):
            H.append([control_H[k, 0], functions[k, 0].f_t])
            H.append([control_H[k, 1], functions[k, 1].f_t])

        U = qt.propagator(
            H,
            t=np.linspace(0, T, len(argsc[0, :, 0]) + 1),
            options=options,
            args={"_step_func_coeff": True},
        )

        return U

    def gate_circuit(self, theta, **kwargs):
        U_timearray = self.propagator(
            argsc=theta, T=self.T, control_H=self.control_H, driving_H=self.driving_H
        )
        return U_timearray[-1].full()

    def full_evolution(self, theta, **kwargs):

        return self.propagator(
            argsc=theta, T=self.T, control_H=self.control_H, driving_H=self.driving_H
        )


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
