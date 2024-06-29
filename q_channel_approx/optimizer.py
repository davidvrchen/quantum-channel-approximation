import time
from typing import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from q_channel_approx.channel import channel_fac, evolver_fac
from q_channel_approx.unitary_circuits import Circuit
from q_channel_approx.training_data import TrainingData, measure_rhoss


def optimize(
    circuit: Circuit,
    training_data: TrainingData,
    max_count: int,
    theta_init: np.ndarray = None,
    n_grad: int = None,
    seed: int = None,
    gamma: float = 10 ** (-4),
    sigmastart: int = 10,
    epsilon: float = 10 ** (-10),
    h: float = 1e-4,
    thread_gradient=False,
    verbose: bool = False,
):

    # change some variable names for more readable code
    _, P = circuit.qubit_layout, circuit.P
    N = training_data.N
    L = training_data.L
    K = training_data.K

    # Set armijo parameters
    sigmabig, sigmasmall, sigmastart = 0, 0, sigmastart
    sigmas = (sigmabig, sigmasmall, sigmastart)
    zero_grad = False

    # initialize some variables
    theta = np.ones(P) if theta_init is None else theta_init

    thetas = np.zeros((max_count, P))
    errors = np.ones(max_count)
    grad_size = np.zeros(max_count)

    # create the helper functions
    N_step_evolver = evolver_fac(circuit=circuit, N=N)

    J = J_fac(N, K, L, N_step_evolver, training_data)

    armijo_update = armijo_update_fac(J=J, gamma=gamma)

    # take care of seeding the optimization indices in case no seed was provided
    random_rng = np.random.default_rng()
    if seed is None:
        seed = random_rng.integers(10**5)
        print(f"optimizer (optimization indices): setting {seed=}")

    rng = np.random.default_rng(seed=seed)
    if n_grad is None:
        n_grad = P
    n_grad = n_grad if n_grad <= P else P

    # create desired gradient function
    if thread_gradient:
        gradient = threaded_gradient_fac(J=J, rng=rng, n_grad=n_grad, P=P, h=h)
    else:
        gradient = gradient_fac(J=J, rng=rng, n_grad=n_grad, P=P, h=h)

    # Set timing parameters
    time_grad = 0
    time_armijo = 0
    time_start = time.time()

    for i in range(max_count):

        time0 = time.time()

        error = J(theta)
        grad = gradient(theta, error)

        thetas[i] = theta.copy()
        errors[i] = error
        grad_size[i] = np.sum(grad * grad)

        time1 = time.time()
        time_grad += time1 - time0

        theta, sigmas, zero_grad = armijo_update(theta, grad, error, sigmas)

        time2 = time.time()
        time_armijo += time2 - time1

        if i % 10 == 0 and verbose:
            print(
                f"""Iteration: {i} \r
            Max gradient term: {np.amax(grad)} \r
            Current gradient: {grad} \r
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


def J_fac(
    N: int,
    K: int,
    L: int,
    N_step_evolver: Callable[[np.ndarray], np.ndarray],
    training_data: TrainingData,
):
    norm_const = N * K * L
    rho0s = training_data.rho0s
    Os = training_data.Os
    Esss = training_data.Esss
    dims_A = training_data.dims_A

    def J(theta):
        rhohatss = np.zeros((L, N + 1, dims_A, dims_A), dtype=np.complex128)
        evolve = N_step_evolver(theta)
        for l in range(L):
            rhohatss[l, :, :, :] = evolve(rho0s[l])

        Ehatsss = measure_rhoss(rhohatss, Os)

        tracesss = Esss - Ehatsss
        return np.sum(tracesss * tracesss) / norm_const

    return J


def armijo_update_fac(J, gamma):

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

    return armijo_update


def gradient_fac(J, rng, n_grad, P, h):
    def gradient(theta, error):

        optimization_ind = rng.choice(P, size=n_grad, replace=False)

        grad_theta = np.zeros(theta.shape)

        for i in range(n_grad):
            theta_p = theta.copy()
            # theta_m = theta.copy()
            theta_p[optimization_ind[i]] = theta_p[optimization_ind[i]] + h
            # theta_m[optimization_ind[i]] = theta_m[optimization_ind[i]] - h

            grad_theta[optimization_ind[i]] = np.real(J(theta_p) - error) / (h)

        return grad_theta

    return gradient


def threaded_gradient_fac(J, rng, n_grad, P, h):
    def threaded_gradient(theta, error):

        optimization_ind = rng.choice(P, size=n_grad, replace=False)

        grad_theta = np.zeros(theta.shape)

        def partial_grad(i):
            theta_p = theta.copy()
            # theta_m = theta.copy()
            theta_p[optimization_ind[i]] = theta_p[optimization_ind[i]] + h
            # theta_m[optimization_ind[i]] = theta_m[optimization_ind[i]] - h

            grad_theta[optimization_ind[i]] = np.real(J(theta_p) - error) / (2 * h)

        opt_range = range(n_grad)

        with ThreadPoolExecutor() as executor:
            executor.map(partial_grad, opt_range)

        return grad_theta

    return threaded_gradient
