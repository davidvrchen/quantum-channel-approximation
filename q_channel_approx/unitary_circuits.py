import itertools
from operator import add
from typing import Callable, NamedTuple

import numpy as np
import scipy as sc

from q_channel_approx.qubit_layouts import QubitLayout
from q_channel_approx.gate_operations import (
    H_fac,
    rx,
    ryd_vdw_fac,
    ryd_dipole_fac,
    xy_fac,
    rz,
    matmul_l,
    CNOT_fac,
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


def unitary_circuit_fac(
    qubit_layout: QubitLayout, operations, repeats: int = 1
) -> Circuit:

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
                return H_fac(H, dims_AB), 0
            case "ryd-vdw", _:
                return ryd_vdw_fac(connections, dims_AB), 1
            case "ryd-dipole", _:
                return ryd_dipole_fac(connections, dims_AB), 1
            case "xy", _:
                return xy_fac(connections, dims_AB), len(connections)
            case "cnot", _:
                return CNOT_fac(connections, dims_AB), 0
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

        return np.linalg.matrix_power(U, repeats)

    return Circuit(unitary, qubit_layout, P, operations)


def HEA_fac(qubit_layout: QubitLayout, depth: int, ent_type: str = "cnot") -> Circuit:
    operations = [
        ("rz", "AB"),
        ("rx", "AB"),
        ("rz", "AB"),
        (ent_type, "AB"),
    ] * depth

    return unitary_circuit_fac(qubit_layout, operations)


def SHEA_trot_fac(
    qubit_layout: QubitLayout,
    H: np.ndarray,
    t: float,
    depth: int,
    q: int,
    ent_type: str = "cnot",
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

    operations = [("ham", (H, t / depth))] + [
        ("rz", "AB"),
        ("rx", "AB"),
        ("rz", "AB"),
        (ent_type, "AB"),
    ] * q

    return unitary_circuit_fac(qubit_layout, operations, repeats=depth)


def SHEA_fac(
    qubit_layout: QubitLayout,
    H: np.ndarray,
    t: float,
    depth: int,
    ent_type: str = "cnot",
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

    operations = [("ham", (H, t))] + [
        ("rz", "AB"),
        ("rx", "AB"),
        ("rz", "AB"),
        (ent_type, "AB"),
    ] * depth

    return unitary_circuit_fac(qubit_layout, operations)
