import numpy as np
import qutip as qt
import scipy as sc


def kron_gates_l(single_gates):
    result = single_gates[0]
    for gate in single_gates[1:]:
        result = np.kron(result, gate)

    return result


def kron_neighbours_even(single_gates):

    l, dims, _ = single_gates.shape
    double_gates = np.zeros((l // 2, dims**2, dims**2), dtype=np.complex128)

    for i in range(0, l // 2):
        double_gates[i, :, :] = np.kron(single_gates[i * 2], single_gates[i * 2 + 1])

    return double_gates


def kron_gates_r(single_gates):
    """Recursively multiply the neighbouring gates.
    When the block size gets below the turnover point the linear
    kron_gates_l is used as it is more efficient in this usecase."""
    TURNOVER = 3

    l = len(single_gates)

    if l > TURNOVER:
        if l % 2 == 0:
            return kron_gates_r(kron_neighbours_even(single_gates))
        return np.kron(
            kron_gates_r(kron_neighbours_even(single_gates[:-1])),
            single_gates[-1],
        )

    return kron_gates_l(np.array(single_gates))


def rz(theta):
    zero = np.zeros(theta.shape)
    exp_m_theta = np.exp(-1j * theta / 2)
    exp_theta = np.exp(1j * theta / 2)

    single_gates = np.einsum(
        "ijk->kij", np.array([[exp_m_theta, zero], [zero, exp_theta]])
    )

    u_gates = kron_gates_l(single_gates)

    return u_gates


def rx(theta):
    costheta = np.cos(theta / 2)
    sintheta = np.sin(theta / 2)

    single_gates = np.einsum(
        "ijk->kij", np.array([[costheta, -sintheta], [sintheta, costheta]])
    )

    u_gates = kron_gates_l(single_gates)

    return u_gates


def H_fac(H, dims_AB):

    if isinstance(H, qt.Qobj):
        H = H.full()

    dims, _ = H.shape
    dims_expand = dims_AB // dims

    def U(t):
        e_H = sc.linalg.expm((-1j) * t * H)
        e_H_exp = np.kron(e_H, np.identity(dims_expand))

        return e_H_exp

    return U


def H_fix_t_fac(H, dims_AB):

    H, t = H

    if isinstance(H, qt.Qobj):
        H = H.full()

    dims, _ = H.shape
    dims_expand = dims_AB // dims

    def U(
        foo,
    ):  # needs a throwaway argument because we are going to pass an empty array in the unitary_fac
        e_H = sc.linalg.expm((-1j) * t * H)
        e_H_exp = np.kron(e_H, np.identity(dims_expand))

        return e_H_exp

    return U


def ryd_ent_fac(connections, dims_AB):

    rydberg = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    n_qubits = dims_AB.bit_length() - 1

    rydberg_2gate = qt.Qobj(rydberg, dims=[[2] * 2, [2] * 2])
    rydberg_gate = np.zeros([dims_AB, dims_AB], dtype=np.complex128)
    for connection in connections:

        id1, id2, d = connection
        ham = qt.expand_operator(
            oper=rydberg_2gate, dims=[2] * n_qubits, targets=[id1, id2]
        ).full()
        rydberg_gate += ham / d**3  # distance to the power -6

    def ryd_ent(theta):
        return sc.linalg.expm(-1j * theta * rydberg_gate)

    return ryd_ent


def xy_ent_fac(connections, dims_AB):
    rydberg = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    n_qubits = dims_AB.bit_length() - 1
    rydberg_2gate = qt.Qobj(rydberg, dims=[[2] * 2, [2] * 2])
    rydberg_gate = np.zeros([dims_AB, dims_AB], dtype=np.complex128)
    for connection in connections:

        id1, id2, d = connection
        ham = qt.expand_operator(
            oper=rydberg_2gate, dims=[2] * n_qubits, targets=[id1, id2]
        ).full()
        rydberg_gate += ham / d**3  # distance to the power -6

    def ryd_ent(theta):

        return sc.linalg.expm(-1j * theta * rydberg_gate)

    return ryd_ent


def matmul_acc_ul(Us: np.ndarray) -> np.ndarray:

    w, dims, _ = Us.shape

    U_lower = np.zeros((w, dims, dims), dtype=np.complex128)
    U_upper = np.zeros((w, dims, dims), dtype=np.complex128)

    U_l_acc = np.identity(dims)
    U_u_acc = np.identity(dims)

    for i, U in enumerate(Us):
        U_l_acc = U_l_acc @ U
        U_lower[i, :, :] = U_l_acc

    for i, U in enumerate(Us[::-1]):
        U_u_acc = U @ U_u_acc
        U_upper[-i - 1, :, :] = U_u_acc

    return U_lower, Us, U_upper


def matmul_acc(Us: np.ndarray) -> np.ndarray:
    Ul, Us, Uu = matmul_acc(Us)
    U = Ul[-1]
    return U


def matmul_l(Us: np.ndarray) -> np.ndarray:
    U_acc = Us[0]

    for U in Us[1:]:
        U_acc = U @ U_acc

    return U_acc
