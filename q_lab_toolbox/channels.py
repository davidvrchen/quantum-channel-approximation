"""
Provides classes that define various trainable quantum channels.


References:
    original code for decay_examply by @lviss

Info:
    Created on Tue March 19 2024

    Last update on Fri Apr 5 2024

    @author: davidvrchen
"""

import math
import random as rd
import time

import numpy as np
import qutip as qt


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
        self, circuit: GateBasedUnitaryCircuit, n_grad_direction: isinstance
    ) -> None:
        """to be documented
        the idea is that this class can find optimal theta for a given circuit
        """

        # related to the circuit
        self.circuit = circuit
        self.m = circuit.m

        # related to the optimization
        self.n_grad_directions = n_grad_direction

        # Set up time variables
        self.time_circuit = 0

    def optimize_theta(self, training_data=None):
        """optimize_theta :: training_data -> optim theta

        Optimizes theta based on the training data.
        Note that the error type and circuit are part of
        the (subclass) GateBasedChannel."""

        optim_theta = self.run_armijo(training_data=training_data, max_count=1000)

        return optim_theta

    @time_wrapper
    def find_gradient(self, theta, training_data, eps=0.01):
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
                print(f"component {i} gives a nan", theta_p[i], theta_m[i])
            grad_theta[i] = (
                self.circuit.J(theta_p, training_data)
                - self.circuit.J(theta_m, training_data)
            ) / (2 * eps)
            theta_p[i] = theta_p[i] - eps
            theta_m[i] = theta_m[i] + eps

            return grad_theta

    def _armijo_update(
        self, flat_theta, training_data, sigmas, grad_theta, gamma=10 ** (-4)
    ):
        """
        to be updated
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
        fid = self.circuit.J(flat_theta, training_data)
        first = True

        # Armijo stepsize rule update
        grad_zero = False

        while not descended:

            update_theta = flat_theta - sigma * grad_theta

            update_fid = self.circuit.J(flat_theta, training_data)

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
        training_data,
        max_count,
        gamma=10 ** (-4),
        sigmastart=1,
        epsilon=0.01,
        save_pulses=True,
    ):
        """
        Function to run the full armijo gradient descend.
        solution saved as self.theta_opt
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

        flat_theta = self.circuit.init_flat_theta()

        # Run update steps
        count = 1
        grad_zero = False
        while count < max_count and not grad_zero and error[count - 1] > 10 ** (-10):

            time0 = time.time()

            error[count] = self.circuit.J(flat_theta, training_data)

            grad_theta = self.find_gradient(flat_theta, training_data, eps=epsilon)
            grad_size[count] = np.inner(np.ravel(grad_theta), np.ravel(grad_theta))

            time1 = time.time()
            time_grad += time1 - time0

            flat_theta, sigmas, grad_zero = self._armijo_update(
                flat_theta, training_data, sigmas, grad_theta, gamma
            )

            time2 = time.time()
            time_armijo += time2 - time1

            if count % 10 == 0 or count == max_count - 1:
                print("Iteration ", count)
                print("   Max gradient term: ", np.amax(grad_theta))
                print("   Current error: ", error[count])
                print("   Current sigma values: ", sigmas)

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

        return self.circuit.reshape_theta(flat_theta), error

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
