import math
import random as rd
from abc import ABC, abstractmethod
import qutip as qt
import numpy as np
import scipy as sc

from channeler.utils.my_functions import generate_gate_connections


class GateBasedUnitaryCircuit(ABC):

    def __init__(
        self,
        m,
        circuit_type="ryd",
        structure="triangle",
        t_ham=0,
        H=0,
        split_H=False,
        **kwargs,
    ):
        self.m = m
        self.t_ham = t_ham
        self.H = H

        self.split_H = split_H

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

    @abstractmethod
    def init_theta(self):
        """Create initial parameters that U understands.

        Returns:
            np.ndarray: initial theta
        """

    @abstractmethod
    def U(self, theta):
        """Returns"""

    def make_channel(self, theta):

        def phi_prime(rho):
            m = rho.shape[0]

            rho_ext = qt.expand_operator(oper=rho, N=3 * m, targets=range(m))
            system = self.U(self.theta) * rho_ext * self.U(self.theta).dag()

            return system.ptrace(range(m))

        return phi_prime


class HardwareAnsatz(GateBasedUnitaryCircuit):
    def U(self, theta):

        depth, m = theta[:, :, 0].shape

        # to parametrize the entanglement
        gate_par = self.set_gate_par(depth, gate_par)

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


from channeler.target_system.settings import TargetSystemSettings



from dataclasses import dataclass


@dataclass
class GateBasedChannelSettings:
    unitary_circuit: GateBasedUnitaryCircuit, 
    target_settings: TargetSystemSettings



class GateBasedChannel:

    def __init__(
        self, channel_s: GateBasedChannelSettings,
    ) -> None:
        """Create a gate based quantum channel.

        Args:
            U (GateBasedUnitaryCircuit): circuit choice for Stinespring unitary
        """
        self.channel_s = channel_s

        self.m = channel_s.target_settings.m

    def optimize_theta(self, training_data):

        theta = self.U.init_theta()

        self.run_armijo(theta=theta, max_count=10)

    def _armijo_update(self, theta, sigmas, grad_theta, gamma=10 ** (-4)):
        """
        Run a single armijo step for a given set of parameters and gradient

        Parameters
        ----------
        theta : np.array
            parameters of unitary circuit
        sigmas : tuple of floats
            (sigmabig, sigmasmall, sigmastart) to iteratively determine an
            optimal starting stepsize.
        grad_theta : np.array
            gradient in the parameters.
        gamma : float, optional
            armijo parameter. The default is 10**(-4).

        Returns
        -------
        update_theta : np.array
            Optimal updated theta.
        sigmas : tuple of floats
            Updated tuple of sigmas.
        grad_zero : bool
            Bool to signal vanishing gradient.

        """

        (sigmabig, sigmasmall, sigmastart) = sigmas

        if sigmabig >= 3:  # Reduce initial step size if consistently to big
            sigmastart = sigmastart / 2
            sigmabig = 0
        if sigmasmall >= 3:  # Increase initial step size if consistently to small
            sigmastart = sigmastart * 2
            sigmasmall = 0

        # Initialize inner loop parameters
        descended = False
        sigma = sigmastart
        fid = self.training_error(theta)
        first = True

        # Armijo stepsize rule update
        grad_zero = False

        # =============================================================================
        #         # Add white noise
        #         max_grad_term = np.amax(grad_theta)
        #         white_noise = 0.05 *max_grad_term *np.random.normal(size = grad_theta.shape)
        #         noise_factor = 1 + white_noise
        #         #grad_theta = grad_theta*noise_factor
        #         #grad_theta = grad_theta + white_noise
        # =============================================================================

        while not descended:

            update_theta = theta - sigma * grad_theta

            update_fid = self.training_error(update_theta, weights=self.weights)

            if update_fid - fid < -(
                gamma * sigma * np.sum(np.multiply(grad_theta, grad_theta))
            ):
                descended = True
                if first:
                    sigmasmall = sigmasmall + 1
            elif sigma < 10**-10:  # or update_fid - fid ==0:
                descended = True
                print("small sigma")
                grad_zero = True
                # count = max_count-1
            else:
                sigma = sigma / 2
                if first:
                    sigmabig = sigmabig + 1
            first = False

        # update_theta = update_theta*noise_factor
        # update_theta = update_theta + white_noise

        return update_theta, (sigmabig, sigmasmall, sigmastart), grad_zero

    def run_armijo(
        self,
        theta,
        max_count,
        gamma=10 ** (-4),
        sigmastart=1,
        epsilon=0.01,
        save_pulses=True,
    ):
        """
        Function to run the full armijo gradient descend.
        solution saved as self.theta_opt

        Parameters
        ----------
        theta : np.array
            initial parameters
        max_count : int
            Max gradient steps.
        gamma : float, optional
            armijo step parameter. The default is 10**(-4).
        sigmastart : float, optional
            initial step size. The default is 1.
        epsilon : float, optional
            step size for finite difference for gate based. The default is 0.01.

        Returns
        -------
        None.

        """
        error = np.ones([max_count])
        grad_size = np.zeros(max_count)

        # Set armijo parameters
        sigmabig = 0
        sigmasmall = 0
        sigmastart = sigmastart
        sigmas = (sigmabig, sigmasmall, sigmastart)

        # Set timing parameters
        time_grad = 0
        time_armijo = 0
        time_start = time.time()

        # Run update steps
        count = 1
        grad_zero = False
        while count < max_count and not grad_zero and error[count - 1] > 10 ** (-10):

            error[count] = self.training_error(
                theta, weights=0, error_type="pauli trace", incl_lambda=False
            )

            time0 = time.time()

            grad_theta = self.find_gradient(theta, eps=epsilon)
            grad_size[count] = np.inner(np.ravel(grad_theta), np.ravel(grad_theta))

            time1 = time.time()
            time_grad += time1 - time0

            theta, sigmas, grad_zero = self._armijo_update(
                theta, sigmas, grad_theta, gamma
            )
            self.theta_opt = theta

            time2 = time.time()
            time_armijo += time2 - time1

            if count % 10 == 0 or count == max_count - 1:
                print("Iteration ", count)
                print("   Max gradient term: ", np.amax(grad_theta))
                print("   Current error: ", error[count])
                print("   Current sigma values: ", sigmas)

                theta1, _ = self.reshape_theta_phi(np.array(theta))

            count += 1
        print("-----")
        print(
            "Grad calculation time: ",
            time_grad,
            " Armijo calculation time: ",
            time_armijo,
        )
        print(
            "Total grad descend time: {}:{:.2f}".format(
                int((time2 - time_start) // 60), ((time2 - time_start) % 60)
            )
        )

        if count < max_count:
            error[count:] = 0
            grad_size[count:] = 0

        self.theta_opt = theta
        self.error = error

    def find_gradient(self, theta, eps=0.01):
        """
        Calculates the gradient for a given set of theta

        Parameters
        ----------
        theta : np.array
        eps : float, optional
            parameter change for finite difference for gate based.
            The default is 0.01.

        Returns
        -------
        gradient

        """

        theta_p = theta.copy()
        theta_m = theta.copy()
        grad_theta = np.zeros(theta.shape)

        optimize_indices = range(len(theta))

        for i in optimize_indices:
            theta_p[i] = theta_p[i] + eps
            theta_m[i] = theta_m[i] - eps
            if math.isnan(theta_p[i]) or math.isnan(theta_m[i]):
                print("component {} gives a nan".format(i), theta_p[i], theta_m[i])
            grad_theta[i] = np.real(
                self.training_error(theta_p) - self.training_error(theta_m)
            ) / (2 * eps)
            theta_p[i] = theta_p[i] - eps
            theta_m[i] = theta_m[i] + eps

        return grad_theta









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
