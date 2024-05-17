import time

from numba import jit, prange
import numpy as np

from q_lab_toolbox.type_hints import (
    Theta,
    DensityMatrix,
    ChannelFactory,
    EvolutionFactory,
    LossFunction,
)
from q_lab_toolbox.unitary_circuits import GateBasedUnitaryCircuit
from q_lab_toolbox.training_data import TrainingData


def optimize(
    circuit: GateBasedUnitaryCircuit,
    training_data: TrainingData,
    max_count: int,
    n_grad: int = None,
    gamma: float = 10 ** (-4),
    sigmastart: int = 10,
    epsilon: float = 0.01,
    h: float = 0.1,
) -> Theta:

    # Set armijo parameters
    sigmabig = 0
    sigmasmall = 0
    sigmastart = sigmastart
    sigmas = (sigmabig, sigmasmall, sigmastart)
    zero_grad = False

    # set accumulation parameters
    theta = circuit.init_theta()
    (P,) = theta.shape
    thetas = np.zeros((max_count, P))
    errors = np.ones(max_count)
    grad_size = np.zeros(max_count)

    # create the helper functions
    N = training_data.N
    Ess = training_data.Ess
    Os = training_data.Os
    rho0s = training_data.rho0s
    L, dims_A, _ = training_data.rho0s.shape

    phi = channel_fac(circuit)

    @jit(forceobj=True)
    def N_step_evolver(theta: np.ndarray, rho0: DensityMatrix) -> list[DensityMatrix]:

        rho_acc = rho0
        rhos = [rho0]
        for _ in prange(N):
            rho_acc = phi(theta, rho_acc)
            rhos.append(rho_acc)

        return rhos

    @jit(forceobj=True, parallel=True)
    def J(theta: Theta) -> float:

        rhohatss = np.zeros((L, N + 1, dims_A, dims_A), dtype=np.complex128)
        for l in prange(L):
            rhohatss[l, :, :, :] = N_step_evolver(theta, rho0s[l])
        Ehatss = measure(Os, rhohatss)

        tracess = Ess - Ehatss
        return np.sum(tracess**2)



    @jit(parallel=True, forceobj=True)
    def gradient(theta: Theta, n_grad=n_grad, P=P) -> Theta:

        if n_grad is None:
            optimization_ind = range(P)
            n_grad = P
        else:
            optimization_ind = np.random.randint(0, P, size=n_grad)

        grad_theta = np.zeros(theta.shape)

        for i in prange(n_grad):
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[optimization_ind[i]] = theta_p[optimization_ind[i]] + h
            theta_m[optimization_ind[i]] = theta_m[optimization_ind[i]] - h

            print(J(theta_p), J(theta_m))

            grad_theta[optimization_ind[i]] = (J(theta_p) - J(theta_m)) / (2 * h)

        return grad_theta

    armijo_update = armijo_update_fac(J, gamma=gamma)

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

        theta, sigmas, zero_grad = armijo_update(theta, grad, sigmas)

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


def armijo_update_fac(J: LossFunction, gamma: float = 10 ** (-4)):

    def armijo_update(theta: Theta, grad: Theta, sigmas: tuple[int, int, int]):

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
        fid = J(theta)
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


def channel_fac(circuit: GateBasedUnitaryCircuit) -> ChannelFactory:

    U_fac = circuit.U_fac()
    dims_A = circuit.qubit_layout.dims_A
    dims_B = circuit.qubit_layout.dims_B

    @jit(forceobj=True)
    def phi(theta: np.ndarray, rho: DensityMatrix) -> DensityMatrix:

        U = U_fac(theta)
        U_conj = U.conj()

        rho_AB = np.kron(rho, np.identity(dims_B))


        inner = np.einsum("ab, bc, dc -> ad", U, rho_AB, U_conj)


        return np.trace(
            inner.reshape([dims_A, dims_B, dims_A, dims_B]), axis1=1, axis2=3
        )

    return phi

def measure(Os: np.ndarray, rhoss: np.ndarray) -> np.ndarray:
    return np.einsum("kab, lnba -> lkn", Os, rhoss)
