import time
import threading

import numpy as np

from q_lab_toolbox.unitary_circuits import Circuit
from q_lab_toolbox.training_data import TrainingData, measure_rhoss


def optimize(
    circuit: Circuit,
    training_data: TrainingData,
    max_count: int,
    n_grad: int = None,
    seed: int = None,
    gamma: float = 10 ** (-4),
    sigmastart: int = 10,
    epsilon: float = 10 ** (-10),
    h: float = 0.01,
):

    def armijo_update(
        theta: np.ndarray,
        grad: np.ndarray,
        error: float,
        sigmas: tuple[int, int, int],
    ):

        sigmabig, sigmasmall, sigmastart = sigmas

        if sigmabig >= 3:  # Reduce initial step size if consistently to big
            sigmastart = sigmastart / 2
            sigmabig = 0
        if sigmasmall >= 3:  # Increase initial step size if consistently to small
            sigmastart = sigmastart * 2
            sigmasmall = 0

        # Initialize inner loop parameters
        descended = False
        sigma = sigmastart
        fid = error
        first = True

        # Armijo stepsize rule update
        zero_grad = False

        while not descended:

            update_theta = theta - sigma * grad

            update_fid = J(update_theta)

            if update_fid - fid < -(gamma * sigma * np.sum(np.multiply(grad, grad))):
                descended = True
                if first:
                    sigmasmall = sigmasmall + 1
            elif sigma < 10**-10:  # or update_fid - fid ==0:
                descended = True
                print("small sigma")
                zero_grad = True
            else:
                sigma = sigma / 2
                if first:
                    sigmabig = sigmabig + 1
            first = False

        return update_theta, (sigmabig, sigmasmall, sigmastart), zero_grad

    unitary, qubit_layout, P, operations = circuit
    dims_A = qubit_layout.dims_A
    dims_B = qubit_layout.dims_B

    # Set armijo parameters
    sigmabig, sigmasmall, sigmastart = 0, 0, sigmastart
    sigmas = (sigmabig, sigmasmall, sigmastart)
    zero_grad = False

    # set accumulation parameters
    theta = np.ones(P) * 1.5
    thetas = np.zeros((max_count, P))
    errors = np.ones(max_count)
    grad_size = np.zeros(max_count)

    # create the helper functions
    N = training_data.N
    Ess = training_data.Ess
    Os = np.array(training_data.Os)
    rho0s = training_data.rho0s
    L = training_data.L
    K = training_data.K

    phi = channel_fac(circuit)

    # @jit(forceobj=True)
    def N_step_evolver(theta):

        phi_theta = phi(theta)

        def _evolver(rho0):
            rho_acc = rho0
            rhos = np.zeros((N + 1, dims_A, dims_A), dtype=np.complex128)
            rhos[0, :, :] = rho0
            for n in range(N):
                rho_acc = phi_theta(rho_acc)
                rhos[n, :, :] = rho_acc

            return rhos

        return _evolver

    norm_const = 2 * L * K * N

    # @jit(forceobj=True)
    def J(theta):
        rhohatss = np.zeros((L, N + 1, dims_A, dims_A), dtype=np.complex128)
        evolve = N_step_evolver(theta)
        for l in range(L):
            rhohatss[l, :, :, :] = evolve(rho0s[l])

        Ehatss = measure_rhoss(rhohatss, Os)

        tracess = Ess - Ehatss
        return np.sum(tracess**2) / norm_const

    if seed is None:
        seed = np.random.randint(10**5)
        print(f"optimizer (optimization indices): setting {seed=}")

    # recommended numpy seeding
    rng = np.random.default_rng(seed=seed)

    def gradient(theta, n_grad=n_grad, P=P):

        if n_grad is None:
            optimization_ind = range(P)
            n_grad = P
        else:
            optimization_ind = rng.integers(0, P, size=n_grad)

        grad_theta = np.zeros(theta.shape)


        for i in range(n_grad):
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[optimization_ind[i]] = theta_p[optimization_ind[i]] + h
            theta_m[optimization_ind[i]] = theta_m[optimization_ind[i]] - h

            grad_theta[optimization_ind[i]] = np.real(J(theta_p) - J(theta_m)) / (2 * h)

        return grad_theta

    def gradient_threaded(theta, n_grad=n_grad, P=P):

        if n_grad is None:
            optimization_ind = range(P)
            n_grad = P
        else:
            optimization_ind = rng.integers(0, P, size=n_grad)

        grad_theta = np.zeros(theta.shape)

        def partial_grad(indices):
            for i in indices:
                theta_p = theta.copy()
                theta_m = theta.copy()
                theta_p[optimization_ind[i]] = theta_p[optimization_ind[i]] + h
                theta_m[optimization_ind[i]] = theta_m[optimization_ind[i]] - h

                grad_theta[optimization_ind[i]] = np.real(J(theta_p) - J(theta_m)) / (2 * h)

        opt_range = range(n_grad)

        partial_grad1 = threading.Thread(target=partial_grad, args=(opt_range[:n_grad//2],))
        partial_grad2 = threading.Thread(target=partial_grad, args=(opt_range[n_grad//2:],))

        partial_grad1.start()
        partial_grad2.start()

        partial_grad1.join()
        partial_grad2.join()

        return grad_theta

    # Set timing parameters
    time_grad = 0
    time_armijo = 0
    time_start = time.time()

    for i in range(max_count):

        time0 = time.time()

        error = J(theta)
        grad = gradient(theta)

        thetas[i] = theta.copy()
        errors[i] = error
        grad_size[i] = np.sum(grad * grad)

        time1 = time.time()
        time_grad += time1 - time0

        theta, sigmas, zero_grad = armijo_update(theta, grad, error, sigmas)

        time2 = time.time()
        time_armijo += time2 - time1

        if i % 10 == 0:
            print(
                f"""Iteration: {i} \r
            Current gradient term: {grad} \r
            Current error: {errors[i]} \r
            Current sigma values: {sigmas}"""
            )

        if zero_grad:
            print(f"Zero gradient hit after {i} iterations")
            return theta, errors[:i], thetas[:i]

        if error < epsilon:
            print(f"Error reduced below threshold of {epsilon} after {i} iterations")
            return theta, errors[:i], thetas[:i]

    print(
        f"""-----\r
        Grad calculation time: \r
        {time_grad} \r
        Armijo calculation time: \r
        {time_armijo} \r
        Total grad descend time: \r
        {int((time2 - time_start) // 60)}:{(time2 - time_start) % 60:.2f}"""
    )

    return theta, errors, thetas


def channel_fac(circuit):

    unitary, qubits, P, operations = circuit
    dims_A = qubits.dims_A
    dims_B = qubits.dims_B

    ancilla = np.zeros((dims_B, dims_B))
    ancilla[0, 0] = 1

    # @jit(forceobj=True)
    def phi(theta):

        U = unitary(theta)
        U_dag = np.transpose(U.conj())

        def approx_phi(rho):
            rho_AB = np.kron(rho, ancilla)
            rho_tensor = (U @ rho_AB @ U_dag).reshape(dims_A, dims_B, dims_A, dims_B)
            return np.trace(rho_tensor, axis1=1, axis2=3)

        return approx_phi

    return phi
