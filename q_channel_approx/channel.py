import numpy as np
from q_channel_approx.unitary_circuits import Circuit


def channel_fac(circuit: Circuit):

    unitary, qubits, P, operations = circuit
    dims_A = qubits.dims_A
    dims_B = qubits.dims_B

    ancilla = np.zeros((dims_B, dims_B))
    ancilla[0, 0] = 1

    def phi(theta):

        U = unitary(theta)
        U_dag = np.transpose(U.conj())

        def approx_phi(rho):

            rho_AB = np.kron(rho, ancilla)
            rho_tensor = (U @ rho_AB @ U_dag).reshape(dims_A, dims_B, dims_A, dims_B)
            return np.trace(rho_tensor, axis1=1, axis2=3)

        return approx_phi

    return phi


def evolver_fac(circuit: Circuit, theta_opt: np.ndarray):

    dims_A = circuit.qubit_layout.dims_A
    phi = channel_fac(circuit)(theta=theta_opt)

    def evolve_n_times(n: int, rho):
        rho_acc = rho.full()
        rhos = np.zeros((n + 1, dims_A, dims_A), dtype=np.complex128)
        rhos[0, :, :] = rho_acc
        for i in range(1, n+1):
            rho_acc = phi(rho_acc)
            rhos[i, :, :] = rho_acc

        return np.array(rhos)

    return evolve_n_times
