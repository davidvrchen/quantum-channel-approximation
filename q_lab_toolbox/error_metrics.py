from abc import ABC, abstractmethod
import numpy as np


class ErrorType(ABC):

    def pre_process(self, training_data):
        # dims = n, l, matrix
        self.training = training_data
        rho_list = self.training_data[0, :, :, :]
        roots = self.training_data_root
        m = self.m
        t_repeats, n_training_rho = self.training.shape[0:2]
        t_repeats -= 1


        return rho_list, roots, m, t_repeats, n_training_rho, theta, gate_par

    @abstractmethod
    def error(self, theta, training_data):
        """Compute the error J ?"""


class Bures(ErrorType):

    def training_error(self, theta):

        training, rho_list, roots, m, t_repeats, n_training_rho, theta, gate_par

        error = 0
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
        return error


class Measurement(ErrorType):
    def error(self, theta, training_data):
        training, rho_list, roots, m, t_repeats, n_training_rho, theta, gate_par
        error = 0
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
                np.random.binomial(self.n_measurements, p) / self.n_measurements * 2 - 1
            )
            error_add += (self.measurements[1, -1, k] - measurement) ** 2
        error += error_add / len(self.paulis)

        return error


class PauliTrace(ErrorType):
    def error(self, theta, training_data):
        training, rho_list, roots, m, t_repeats, n_training_rho, theta, gate_par
        rhos_approx = np.zeros(
            (t_repeats, n_training_rho, 2**m, 2**m), dtype=np.csingle
        )
        for i in range(n_training_rho):
            rhos_approx[:, i] = self.unitary_approx_n(t_repeats, rho_list[i], U)[1:]

        # Old method, better at using all cores, but slower overall
        pauli_rho = np.real(
            np.einsum("nlab, kba -> nlk", rhos_approx, self.paulis, optimize="greedy")
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

        return error


class TraceProduct(ErrorType):
    def error(self):
        training, rho_list, roots, m, t_repeats, n_training_rho, theta, gate_par
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

        error += np.real(error_add) / (n_training_rho * len(self.paulis) * t_repeats)
        return error
