import itertools
from operator import add
from typing import Callable, NamedTuple
from q_channel_approx.qubit_layouts import QubitLayout

import numpy as np

from q_channel_approx.gate_operations import (
    H_fac,
    rx,
    ryd_ent_fac,
    xy_ent_fac,
    rz,
    H_fix_t_fac,
    matmul_l,
)


class Circuit(NamedTuple):
    U: Callable[[np.ndarray], np.ndarray]
    qubit_layout: QubitLayout
    P: int
    operations: list[tuple[str, str | np.ndarray]]

    def __repr__(self) -> str:
        return f"Circuit; qubits layout: \n {self.qubit_layout} \n Parameters: {self.P} \n Operations {self.operations}"

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        return self.U(theta)


def count_qubits(dims: int) -> int:
    return dims.bit_length() - 1


def unitary_circuit_fac(qubit_layout: QubitLayout, operations) -> Circuit:

    dims_A = qubit_layout.dims_A
    dims_AB = qubit_layout.dims_AB
    connections = qubit_layout.gate_connections

    DIMS_MAP = {
        "A": count_qubits(dims_A),
        "B": count_qubits(dims_AB // dims_A),
        "AB": count_qubits(dims_AB),
    }

    def init_gate(operation) -> tuple[Callable[[np.ndarray], np.ndarray], int]:
        match operation:
            case "rz", dims:
                return rz, DIMS_MAP[dims]
            case "rx", dims:
                return rx, DIMS_MAP[dims]
            case "ham", H:
                return H_fac(H, dims_AB), 1
            case "ham fix t", H:
                return H_fix_t_fac(H, dims_AB), 0
            case "ryd ent", _:
                return ryd_ent_fac(connections, dims_AB), 1
            case "xy ent", _:
                return xy_ent_fac(connections, dims_AB), 1
            case _:
                raise ValueError(f"unknown gate: {operation}")

    _operations = [init_gate(operation) for operation in operations]

    D = len(_operations)

    params = [params for gate, params in _operations]
    params_acc = [0] + list(itertools.accumulate(params, add))
    P = sum(params)

    def unitary(theta):

        Us = np.zeros((D, dims_AB, dims_AB), dtype=np.complex128)

        for d, operation in enumerate(_operations):
            gate, params = operation
            Us[d, :, :] = gate(theta[params_acc[d] : params_acc[d + 1]])

        U = matmul_l(Us)

        return U

    return Circuit(unitary, qubit_layout, P, operations)


def HEA_fac(
    qubit_layout: QubitLayout, depth: int, ent_type: str = "ryd ent"
) -> Circuit:
    operations = [
        ("rz", "AB"),
        ("rx", "AB"),
        ("rz", "AB"),
        (ent_type, ""),
    ] * depth

    return unitary_circuit_fac(qubit_layout, operations)


def SHEA_trot_fac(
    qubit_layout: QubitLayout,
    H: np.ndarray,
    t: float,
    depth: int,
    ent_type: str = "ryd ent",
) -> Circuit:
    """Trotterized H, does a small H block for time `t` followed by one HEA cycle (ZXZ, ent)
    This sequence is repeated `depth` times.

    Args:
        qubit_layout (QubitLayout): _description_
        H (np.ndarray): _description_
        t (float): _description_
        depth (int): _description_

    Returns:
        Circuit: _description_
    """

    operations = [
        ("ham fix t", (H, t)),
        ("rz", "AB"),
        ("rx", "AB"),
        ("rz", "AB"),
        (ent_type, ""),
    ] * depth

    return unitary_circuit_fac(qubit_layout, operations)


def SHEA_fac(
    qubit_layout: QubitLayout,
    H: np.ndarray,
    t: float,
    depth: int,
    ent_type: str = "ryd ent",
) -> Circuit:
    """Starts with H block for `t`, them does HEA with `depth`.

    Args:
        qubit_layout (QubitLayout): _description_
        H (np.ndarray): _description_
        t (float): _description_
        depth (int): _description_

    Returns:
        Circuit: _description_
    """

    operations = [("ham fix t", (H, t))] + [
        ("rz", "AB"),
        ("rx", "AB"),
        ("rz", "AB"),
        (ent_type, ""),
    ] * depth

    return unitary_circuit_fac(qubit_layout, operations)
