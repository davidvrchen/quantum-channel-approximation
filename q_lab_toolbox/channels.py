"""
Provides a classes that define various ways trainable quantum channels.


References:
    original code for decay_examply by @lviss

Info:
    Created on Tue March 19 2024

    @author: davidvrchen
"""

import math
import random as rd
import time
from abc import ABC, abstractmethod

import matplotlib.animation as animation
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import scipy as sc
from numpy.core.umath_tests import inner1d

from q_lab_toolbox.error_metrics import ErrorType, Measurement
from q_lab_toolbox.utils.my_functions import (
    Znorm,
    create_control_hamiltonians,
    create_driving_hamiltonians,
    generate_gate_connections,
    get_paulis,
    wasserstein1,
)
from Stinespring_unitary_circuits import (
    U_circuit,
    U_circuit_pulse,
    generate_gate_connections,
)
from q_lab_toolbox.unitary_circuits import GateBasedUnitaryCircuit


class GateBasedChannel:

    def time_wrapper(func):
        """
        Decorator to time class methods.
        Modify the function to time the total function evaluation time and
        count the number of method calls. Data is saved as class paramater as
        'timed_{}' and 'calls_{}' with {} = function name

        Parameters
        ----------
        func : method of class


        Returns
        -------
        func : decorated func

        """

        def innerwrap(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()

            try:
                args[0].__dict__["timed_" + str(func.__name__)] += end - start
            except KeyError:
                args[0].__dict__["timed_" + str(func.__name__)] = end - start

            try:
                args[0].__dict__["calls_" + str(func.__name__)] += 1
            except KeyError:
                args[0].__dict__["calls_" + str(func.__name__)] = 1

            return result

        return innerwrap

    def print_times(self):
        """
        Prints the times and calls recorded by the time_wrapper

        Returns
        -------
        None.

        """
        print("-----")
        for key, val in self.__dict__.items():
            if key.startswith("timed_"):
                print(
                    "Function",
                    key[6:],
                    "took {0:d}:{1:.2f} min:sec".format(int(val // 60), val % 60),
                    end="",
                )
                print(" for {} calls".format(self.__dict__["calls_" + key[6:]]))
        print("-----")

    def __init__(
        self,
        *,
        m,
        par_dict=None,
        circuit: GateBasedUnitaryCircuit,
        error_type: ErrorType,
    ) -> None:
        """Create a gate based quantum channel.

        Args:
            circuit (GateBasedUnitaryCircuit): circuit choice for Stinespring unitary

            error_metric (ErrorMetric):
        """
        if par_dict is None:
            par_dict = {}

        # System settings
        self.m = m
        state_00 = np.zeros([2 ** (self.m + 1), 2 ** (self.m + 1)])
        state_00[0, 0] = 1
        self.state_00 = state_00

        self.weights = 0
        self.steadystate_weight = par_dict["steadystate_weight"]

        # Set up error variables
        self.error_type = error_type

        # Set up circuit variables
        self.circuit = circuit

        self.qubit_structure = par_dict["qubit_structure"]

        self.n_grad_directions = par_dict["n_grad_directions"]
        self.num_gate_pairs = generate_gate_connections(
            self.m, self.qubit_structure, cutoff=par_dict["cutoff"]
        )

        # Set up time variables
        self.time_circuit = 0

    def update_pars(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
            self.__dict__[key] = kwargs[key]

    def run_all_lindblad(
        self,
        max_it_training,
        epsilon=0.01,
        gamma=10 ** (-4),
        sigmastart=1,
        **kwargs,
    ):

        theta_opt, error = self.run_armijo(
            self.circuit.init_flat_theta(),
            max_it_training,
            gamma=gamma,
            sigmastart=sigmastart,
            epsilon=epsilon,
        )

        self.print_times()

        return theta_opt, error

    def run_all_unitary(
        self,
        U,
        n_training,
        seed,
        depth,
        theta0,
        max_it_training,
        epsilon=0.01,
        gamma=10 ** (-4),
        sigmastart=1,
        circuit_type="ryd",
        pauli_type="order 1",
        t_repeated=2,
        **kwargs,
    ):
        self.depth = depth

        self.set_original_unitary(U)

        self.set_training_data(
            n_training, seed, paulis=pauli_type, t_repeated=t_repeated
        )

        self.set_unitary_circuit(depth=depth, circuit_type=circuit_type, **kwargs)

        self.run_armijo(
            theta0, max_it_training, gamma=gamma, sigmastart=sigmastart, epsilon=epsilon
        )

        return self.theta_opt, self.error

    def set_original_lindblad(self, H, An, t_ham):
        """
        Sets self.evolution(rho0) as a function that simulates a lindbladian:
        i d_t rho = [H,rho] + sum_k An[k] rho An[k]* - 1/2 [An[k]*An[k], rho]

        Parameters
        ----------
        H : np.ndarray, 2**m x 2**m
            Hamiltonian acting on the qubits.
        An : np.ndarray, n by 2**m x2**m
            An matrices in the lindbladian.
        t_ham : float
            fixed time to integrate the lindbladian to.

        """
        self.H = H
        self.t_ham = t_ham
        m = self.m

        def lindblad_evolution(t_eval, rho0):

            def lindbladian(t, rho):
                rho = np.reshape(rho, (2**m, 2**m))
                result = -1j * (H @ rho - rho @ H)
                for A in An:
                    Ad = np.conj(np.transpose(A))
                    result = result + A @ rho @ Ad - Ad @ A @ rho / 2 - rho @ Ad @ A / 2
                result = np.reshape(result, 4**m)
                return result

            solver = sc.integrate.complex_ode(lindbladian)
            solver.set_initial_value(np.reshape(rho0, 4**m), 0)

            if type(t_eval) == np.ndarray:
                sol = np.zeros([len(t_eval), 4**m], dtype=np.complex128)
                sol[0] = solver.integrate(t_eval[0])
                for i in range(1, len(t_eval)):
                    solver.set_initial_value(sol[i - 1], t_eval[i - 1])
                    sol[i] = solver.integrate(t_eval[i])
                return np.reshape(sol, [len(t_eval), 2**m, 2**m])

            else:
                sol = solver.integrate(t_eval)
                if solver.successful():
                    return np.reshape(sol, [2**m, 2**m])
                else:
                    print("Solver for lindblad evolution aborted")
                    return rho0

        self.evolution = lambda rho0: lindblad_evolution(t_ham, rho0)
        self.evolution_t = lindblad_evolution
        self.from_lindblad = True

    def set_original_unitary(self, U):
        """
        Sets self.evolution(rho0) as a function that simulates another unitary circuit:
        Phi_t(rho) = Tr_B[U (rho0 x |0><0|) U*]

        Parameters
        ----------
        U : np.ndarray 2**(2*m) x 2**(2*m)
            The unitary matrix on the system with twice the number of qubits.

        """

        def unitary_evolution(rho):
            m = self.m
            UrhoU = U @ np.kron(rho, self.state_00) @ np.transpose(np.conjugate(U))
            return np.trace(
                UrhoU.reshape(2**m, 2 ** (m + 1), 2**m, 2 ** (m + 1)), axis1=1, axis2=3
            )

        self.evolution = unitary_evolution
        self.from_lindblad = False

    @time_wrapper
    def set_training_data(self, n_training, seed, paulis="order 1", t_repeated=2):
        """
        Initialises all the training data as class parameters


        Parameters
        ----------
        n_training : int
            Number of training samples to use.
        seed : int
            Seed for the training samples, used for reproducible data.
        paulis : str, optional
            Determines the type of pauli strings that will be used as observables.
            Options: 'order k', 'full', 'random n'
            The default is 'order 1'.
        t_repeated : int, optional
            Number of repeated timesteps to use as data. The default is 2.

        Returns
        -------
        None.

        """
        m = self.m

        self.set_steady_state()

        # rho_list = np.zeros((n_training,2**m,2**m),dtype = np.csingle)

        # dims = n, l, matrix
        training = np.zeros((t_repeated + 1, n_training, 2**m, 2**m), dtype=np.csingle)
        training_root = np.zeros(
            (t_repeated + 1, n_training, 2**m, 2**m), dtype=np.csingle
        )

        # dims = k, matrix
        paulis, pauli_names, pauli_id_list, pauli_indices = get_paulis(m, space=paulis)

        # dims = n, l, k (time, data, pauli)
        traces = np.zeros((t_repeated + 1, n_training, len(paulis)))
        measurements = np.zeros((t_repeated + 1, n_training, len(paulis)))

        for l in range(n_training):
            if l == 0:
                random_ket = qt.rand_ket_haar(dims=[[2**m], [1]], seed=seed)
                random_ket.dims = [[2] * m, [2] * m]
                random_bra = random_ket.dag()
                rho = (random_ket * random_bra).full()
                np.random.seed(seed)
            elif l == n_training - 1:
                rho = self.steady_state
            else:

                # =============================================================================
                #                 # Pure initialization
                #                 random_ket = qt.rand_ket_haar(dims = [[2**m], [1]], seed = seed)
                #                 random_ket.dims = [[2]*m,[2]*m]
                #                 random_bra = random_ket.dag()
                #                 rho = (random_ket * random_bra).full()
                # =============================================================================

                # Mixed initialization, randomly sets the eigenvalues s.t.
                # sum_i lambda_i = 1
                mix_factor = np.random.rand() ** 1 / 2

                evals = np.random.normal(size=2**m)
                evals = evals**2 / np.sum(evals**2)

                # print("Purity of initial state: {:.2f} with evals \n    {}".format(sum(evals**2), np.sort(evals)))

                # zero matrix
                zero_mat = np.zeros((2**m, 2**m))
                zero_mat[0, 0] = 1

                # mixed matrix
                init_matrix = mix_factor * zero_mat + (1 - mix_factor) * np.diag(evals)
                random_mixed = qt.Qobj(init_matrix, dims=[[2] * m, [2] * m])

                U = qt.random_objects.rand_unitary_haar(N=2**m, dims=[[2] * m, [2] * m])
                rho = (U * random_mixed * U.dag()).full()

            training[:, l, :, :] = np.reshape(
                self.evolution_n(t_repeated, rho), (t_repeated + 1, 2**m, 2**m)
            )

            for t_ind in range(t_repeated + 1):
                training_root[t_ind, l, :, :] = sc.linalg.sqrtm(
                    training[t_ind, l, :, :]
                )
                for k, pauli in enumerate(paulis):
                    traces[t_ind, l, k] = np.real(
                        np.trace(training[t_ind, l, :, :] @ pauli)
                    )
                    if isinstance(self.error_type, Measurement):
                        prob = min(max((traces[t_ind, l, k] + 1) / 2, 0.0), 1.0)
                        measurements[t_ind, l, k] = (
                            np.random.binomial(self.n_measurements, prob)
                            / self.n_measurements
                            * 2
                            - 1
                        )

        # self.training_rho = rho_list
        self.training_data = training
        self.training_data_root = training_root

        self.traces = traces
        self.measurements = measurements

        self.paulis = paulis
        self.pauli_names = pauli_names
        self.pauli_id_list = pauli_id_list
        self.pauli_indices = pauli_indices

    @time_wrapper
    def training_error(
        self, flat_theta, weights=0, error_type="internal", incl_lambda=True
    ):
        """
        Determines the error of the circuit for a given set of parameters and a
        given error type

        Parameters
        ----------
        theta_phi : np.array, 1 dimensional
            Single list of all parameters.
        weights : float or np.array, optional
            list of weights for wasserstein error (currently not implemented).
            The default is 0, which calculates the weights internally.
        error_type : str, optional
            Type of error to use. The default is 'internal' to streamline error
            type throughout the code.

        Returns
        -------
        error : float
            Total error.

        """
        if error_type == "internal":
            error_type = self.error_type

        # dims = n, l, matrix
        training = self.training_data
        rho_list = self.training_data[0, :, :, :]
        roots = self.training_data_root
        m = self.m
        t_repeats, n_training_rho = training.shape[0:2]
        t_repeats -= 1

        theta = self.circuit.reshape_theta(flat_theta=flat_theta)

        time0 = time.time()
        U = self.circuit.U(theta=theta)
        time1 = time.time()
        self.time_circuit += time1 - time0

        error = 0
        if error_type == "bures":
            for i in range(n_training_rho - 1):
                rhos_approx = self.unitary_approx_n(t_repeats, rho_list[i], U)
                for nt in range(1, t_repeats + 1):
                    error += max(
                        0,
                        1
                        - np.abs(
                            np.trace(
                                sc.linalg.sqrtm(
                                    np.einsum(
                                        "ij, jk, kl",
                                        roots[nt, i],
                                        rhos_approx[nt],
                                        roots[nt, i],
                                        optimize="greedy",
                                    )
                                )
                            )
                        ),
                    )

            error = error / ((n_training_rho - 1) * t_repeats)

            steadystate_approx = self.unitary_approx_n(1, rho_list[-1], U)[1]
            error += self.steadystate_weight * max(
                0,
                1
                - np.abs(
                    np.trace(
                        sc.linalg.sqrtm(
                            np.einsum(
                                "ij, jk, kl",
                                roots[1, -1],
                                steadystate_approx,
                                roots[1, -1],
                                optimize="greedy",
                            )
                        )
                    )
                ),
            )

        elif error_type == "trace":
            for i in range(n_training_rho - 1):
                rhos_approx = self.unitary_approx_n(t_repeats, rho_list[i], U)
                for nt in range(1, t_repeats + 1):
                    error += (
                        np.vdot(
                            (rhos_approx[nt] - training[nt, i]).T,
                            rhos_approx[nt] - training[nt, i],
                        )
                        ** (1 / 2)
                        / 2
                    )

            error = np.real(error) / ((n_training_rho - 1) * t_repeats)

            steadystate_approx = self.unitary_approx_n(1, rho_list[-1], U)[1]
            error += (
                self.steadystate_weight
                * np.vdot(
                    (steadystate_approx - training[1, -1]).T,
                    steadystate_approx - training[1, -1],
                )
                ** (1 / 2)
                / 2
            )

        elif error_type == "pauli trace":
            rhos_approx = np.zeros(
                (t_repeats, n_training_rho, 2**m, 2**m), dtype=np.csingle
            )
            for i in range(n_training_rho):
                rhos_approx[:, i] = self.unitary_approx_n(t_repeats, rho_list[i], U)[1:]

            # Old method, better at using all cores, but slower overall
            pauli_rho = np.real(
                np.einsum(
                    "nlab, kba -> nlk", rhos_approx, self.paulis, optimize="greedy"
                )
            )

            # pauli_rho = np.sum(np.real(rhos_approx[:,:,self.pauli_indices[1],self.pauli_indices[0]]*self.pauli_indices[2]),axis = -1)

            error = (self.traces[1:, :, :] - pauli_rho) ** 2

            # full steady state
            error[:, -1, :] = error[:, -1, :] * self.steadystate_weight

            # =============================================================================
            #             # steady state for 1 time step
            #             error[2:,-1,:] = error[2:,-1,:]*0
            #             error[1,-1,:] = error[1,-1,:]*self.steadystate_weight
            # =============================================================================

            error = np.einsum("nlk ->", error, optimize="greedy") / (
                2 * n_training_rho * len(self.paulis) * t_repeats
            )
            error = max(0, np.real(error))

        elif error_type == "trace product":
            rhos_approx = np.zeros(
                (t_repeats, n_training_rho - 1, 2**m, 2**m), dtype=np.csingle
            )
            for i in range(n_training_rho - 1):
                rhos_approx[:, i] = self.unitary_approx_n(t_repeats, rho_list[i], U)[1:]
            error = -np.einsum(
                "nlk, kij, nlji ->", self.traces[1:, 0:-1], self.paulis, rhos_approx
            )
            error = error / (n_training_rho * len(self.paulis) * t_repeats)
            error = np.real(error)

            steadystate_approx = self.unitary_approx_n(1, rho_list[-1], U)[1]
            error_add = -np.einsum(
                "k, kij, ji-> ", self.traces[1, -1], self.paulis, steadystate_approx
            )

            error += np.real(error_add) / (
                n_training_rho * len(self.paulis) * t_repeats
            )

        elif error_type == "measurement":
            for l in range(n_training_rho - 1):
                rhos_approx = self.unitary_approx_n(t_repeats, rho_list[i], U)
                for nt in range(1, t_repeats + 1):
                    for k in range(len(self.paulis)):
                        trace = np.real(np.trace(rhos_approx[nt] @ self.paulis[k]))
                        p = max(min((trace + 1) / 2, 1), 0)
                        measurement = (
                            np.random.binomial(self.n_measurements, p)
                            / self.n_measurements
                            * 2
                            - 1
                        )
                        error += (self.measurements[nt, l, k] - measurement) ** 2
            error = error / ((n_training_rho - 1) * len(self.paulis) * t_repeats)
            error = max(0, error)

            steadystate_approx = self.unitary_approx_n(1, rho_list[-1], U)[1]
            error_add = 0
            for k in range(len(self.paulis)):
                trace = np.real(np.trace(steadystate_approx @ self.paulis[k]))
                p = max(min((trace + 1) / 2, 1), 0)
                measurement = (
                    np.random.binomial(self.n_measurements, p) / self.n_measurements * 2
                    - 1
                )
                error_add += (self.measurements[1, -1, k] - measurement) ** 2
            error += error_add / len(self.paulis)

        elif error_type == "wasserstein":
            # =============================================================================
            #             calc_weights = False
            #             if type(weights)!= np.ndarray:
            #                 weights = np.zeros([t_repeats,n_training_rho, len(self.paulis)])
            #                 calc_weights = True
            #
            #             for i in range(n_training_rho):
            #                 rhos_approx = self.unitary_approx_n(n_training_rho, rho_list[i], U)
            #
            #                 for nt in range(1,t_repeats+1):
            #                     if calc_weights:
            #                         _, weights[nt-1,i,:] = wasserstein1(rhos_approx[nt], self.training_data[nt,i], (self.paulis, self.pauli_id_list))
            #
            #                     for j in range(len(self.paulis)):
            #                         #error += np.trace(weights[nt,i,j]* self.paulis[j] @ (rhos_approx[nt] - self.training_data[nt,i]))
            #                         error += weights[nt-1,i,j]*np.sum(inner1d(self.paulis[j].T, rhos_approx[nt] - self.training_data[nt,i]))
            #             error = np.real(error/(n_training_rho))
            #             if calc_weights:
            #                 self.weights = weights
            # =============================================================================

            calc_weights = False
            if type(weights) != np.ndarray:
                weights = np.zeros(
                    [
                        len(self.paulis),
                    ]
                )
                calc_weights = True

            rhos_approx = np.zeros(
                (t_repeats, n_training_rho - 1, 2**m, 2**m), dtype=np.csingle
            )
            for i in range(n_training_rho - 1):
                rhos_approx[:, i] = self.unitary_approx_n(t_repeats, rho_list[i], U)[1:]

            rhos_approx_sum = np.sum(rhos_approx, axis=(0, 1))
            rhos_exact_sum = np.sum(training[1:], axis=(0, 1))

            if calc_weights:
                _, weights = wasserstein1(
                    rhos_approx_sum, rhos_exact_sum, (self.paulis, self.pauli_id_list)
                )
                self.weights = weights

            error = np.einsum(
                "k, kab, ba ->", weights, self.paulis, rhos_approx_sum - rhos_exact_sum
            )
            error = np.real(error)

        elif error_type == "rel entropy":
            pass

        else:
            print(f"Error type {self.error_type} not found")

        return error

    @time_wrapper
    def find_gradient(self, theta, eps=0.01):
        """
        Calculates the gradient for a given set of theta

        Parameters
        ----------
        theta : np.array
            for pulse based:
                - dims n_controls x Zdt x 2.
            for gate based:
                - 1 dimensional, length dependent on gate parameters
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

        if self.n_grad_directions != -1:
            optimize_indices = rd.sample(
                list(range(len(theta))), self.n_grad_directions
            )
        else:
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
        flat_theta,
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
                flat_theta, weights=0, error_type="pauli trace", incl_lambda=False
            )

            time0 = time.time()

            grad_theta = self.find_gradient(flat_theta, eps=epsilon)
            grad_size[count] = np.inner(np.ravel(grad_theta), np.ravel(grad_theta))

            time1 = time.time()
            time_grad += time1 - time0

            flat_theta, sigmas, grad_zero = self._armijo_update(
                flat_theta, sigmas, grad_theta, gamma
            )
            theta_opt = flat_theta

            time2 = time.time()
            time_armijo += time2 - time1

            if count % 10 == 0 or count == max_count - 1:
                print("Iteration ", count)
                print("   Max gradient term: ", np.amax(grad_theta))
                print("   Current error: ", error[count])
                print("   Current sigma values: ", sigmas)

                theta1, _ = self.circuit.reshape_theta(flat_theta)

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

        return flat_theta, error

    def evolution_n(self, n, rho):
        """
        Calculate the exact evolution for n time steps of rho

        Parameters
        ----------
        n : int
            number of time steps.
        rho : np.array, 2**m x 2**m
            density matrix.

        Returns
        -------
        rho_end : np.array, n+1 x 2**m x 2**m
            list of evolutions of rho.

        """
        rho_end = np.zeros((n + 1, 2**self.m, 2**self.m), dtype=np.complex128)
        rho_end[0] = rho
        for i in range(n):
            rho = self.evolution(rho)
            rho_end[i + 1] = rho
        return rho_end

    @time_wrapper
    def unitary_approx_n(self, n, rho, U=np.array([None])):
        """
        Get Phi_kt(rho) = Phi_t(Phi_t(...Phi_t(rho)...) for k = 0 till k = n

        Parameters
        ----------
        n : int
            Number of repetitions of the circuit.
        rho : np.array, 2**m x 2**m
            density matrix to start with.
        U : np.array, 2**(2*m+1) x 2**(2*m+1)
            Unitary evolution on the bigger system
            Gets internal if not specified

        Returns
        -------
        rho : np.array, n+1 x 2**m x 2**m
            density matrix after [1,...,n] evolutions

        """

        rho_end = np.zeros((n + 1, 2**self.m, 2**self.m), dtype=np.complex128)
        rho_end[0] = rho

        if (U == None).any():
            try:
                theta = self.circuit.reshape_theta(flat_theta=flat_thata)
                U = self.circuit.U(theta=theta)
            except AttributeError:
                print("No optimal theta found and no unitary specified")
                return rho_end

        Udag = np.transpose(np.conjugate(U))
        for i in range(n):
            UrhoU = U @ np.kron(rho, self.state_00) @ Udag
            rho = np.trace(
                UrhoU.reshape(
                    2**self.m, 2 ** (self.m + 1), 2**self.m, 2 ** (self.m + 1)
                ),
                axis1=1,
                axis2=3,
            )
            rho_end[i + 1] = rho

        return rho_end

    def set_steady_state(self):
        """
        set self.steady_state as steady state of the system

        Returns
        -------
        None.

        """
        random_ket = qt.rand_ket_haar(dims=[[2**self.m], [1]])
        random_ket.dims = [[2] * self.m, [2] * self.m]
        random_bra = random_ket.dag()
        steady_state_old = (random_ket * random_bra).full()
        steady_state_new = self.evolution(steady_state_old)
        count = 0
        maxcount = 5000
        while (
            np.amax(np.abs(steady_state_old - steady_state_new)) > 10 ** (-6)
            and count < maxcount
        ):
            steady_state_old = steady_state_new
            steady_state_new = self.evolution(steady_state_old)
            count += 1
        if count == maxcount:
            print("Steady state not found")
        self.steady_state = steady_state_new
