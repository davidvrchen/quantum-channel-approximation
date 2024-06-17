import itertools
import threading
from abc import ABC, abstractmethod
from operator import add
from typing import Callable, Iterable, NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from q_channel_approx.gate_operations import (
    H_fac,
    rx,
    ryd_ent_fac,
    rz,
    H_fix_t_fac,
    matmul_l,
)


class Qubit(NamedTuple):
    """Representation of a qubit in a layout, four fields.

    Args:
        x (float): x position of the qubit.
        y (float): y position of the qubit.
        type ("computational", "ancilla"): type of the qubit
        id (int): unique integer id, such that all id's
        in a given layout enumerate the total amount of qubits.
    """

    x: float
    y: float
    type: str
    id: int


class GateConnection(NamedTuple):
    """Representation a connection between 2 qubits, three fields.

    Args:
        id1 (int): id of qubit 1.
        id2 (int): id of qubit 2.
        d (float): distance between the qubits.
    """

    id1: int
    id2: int
    d: float


def enumerate_qubits(triples: Iterable[tuple[float, float, str]]) -> tuple[Qubit]:
    """Assigns a unique id to all triples of kind (x, y, type) and
    converts them into Qubit objects. The first triple gets id=0,
    the id assignment just enumerates the other triples from there.

    Args:
        triples (Iterable[tuple[float, float, str]]): the triples to be converted to
        into qubits with id

    Returns:
        tuple[Qubit]: list of the Qubit objects in a given layout.
    """

    return tuple(Qubit(*triple, id=i) for i, triple in enumerate(triples))


def dist(q1: Qubit, q2: Qubit) -> float:
    """Calculate the (regular Euclidean) distance between two qubits.

    Args:
        q1 (Qubit): a qubit.
        q2 (Qubit): the other qubit.

    Returns:
        float: the distance between the qubits.
    """

    return np.sqrt((q1.x - q2.x) ** 2 + (q1.y - q2.y) ** 2)


class QubitLayout(ABC):
    def __init__(self, m: int, cutoff: float = 1) -> None:

        self.cutoff = cutoff
        self.qubits = self.place_qubits(m)

        self.gate_connections = self.find_gate_connections()

        self.comp_qubits = [
            qubit for qubit in self.qubits if qubit.type == "computational"
        ]
        self.anc_qubits = [qubit for qubit in self.qubits if qubit.type == "ancilla"]

        # calc some values related to the number of
        # computational qubits (m) and ancilla qubits (n_ancilla)
        self.m = m
        self.n_ancilla = self.count_ancillas()

        # calc the dimensionality of the involved Hilbert spaces
        self.dims_A = 2**m
        self.dims_B = 2**self.n_ancilla
        self.dims_AB = self.dims_A * self.dims_B

    def __repr__(self) -> str:
        return f"Qubit layout ({self.m} comp qubits {self.n_ancilla} ancilla qubits)"

    @abstractmethod
    def place_qubits(self, m: int) -> tuple[Qubit]:
        """Return iterable of all the Qubits in the circuit."""

    def count_ancillas(self) -> int:
        """Function that determines the number of ancilla qubits."""
        return sum([1 for qubit in self.qubits if qubit.type == "ancilla"])

    def find_gate_connections(self) -> Iterable[GateConnection]:
        """calculate the distances between all pairs of qubits.
        All qubits that are closer than some threshold value have a
        gate connection (q_id1, q_id2, d)."""

        pairs = itertools.combinations(self.qubits, 2)

        gate_connections = [
            GateConnection(q1.id, q2.id, d)
            for (q1, q2) in pairs
            if (d := dist(q1, q2)) <= self.cutoff
        ]

        return gate_connections

    def show_layout(self, c_map: dict = None) -> Axes:
        """Create (using matplotlib) a visual representation
        of the qubit layout.

        Args:
            c_map (dict, optional): Color mapping for used to
            distinguish computational and ancilla qubits.
            Defaults to None which yields default matplotlib colors.

        Returns:
            Axes: axes object on which the qubit layout is plotted.
        """

        ax = plt.gca()

        OFFSET_X = -0.25
        OFFSET_Y = -0.25

        if c_map is None:
            c_map = {"computational": "#1f77b4", "ancilla": "#ff7f0e"}

        for qubit in self.qubits:
            plt.scatter(
                qubit.x,
                qubit.y,
                c=c_map[qubit.type],
                s=500,
                label=qubit.type,
                edgecolors="none",
            )
            ax.annotate(
                qubit.id,
                (qubit.x + OFFSET_X, qubit.y + OFFSET_Y),
                fontsize=8,
                weight="bold",
            )

        ax.set_aspect("equal")
        ax.margins(x=1, y=1)
        plt.axis("off")

        comp_patch = mpl.patches.Patch(
            color=c_map["computational"], label="Computational"
        )
        anc_patch = mpl.patches.Patch(color=c_map["ancilla"], label="Ancilla")

        patches = [comp_patch, anc_patch] if self.n_ancilla > 0 else [comp_patch]

        plt.legend(handles=patches)

        plt.title("Qubit layout", weight="bold")
        return ax


class TriangularLayoutAB(QubitLayout):
    def __init__(self, m: int, cutoff: float = 1, distance: float = 1) -> None:
        self.distance = distance
        super().__init__(m, cutoff)

    def __repr__(self):
        return f"Triangular qubit layout ({self.m} comp. qubits, {self.n_ancilla} ancilla qubits)"

    def place_qubits(self, m: int) -> tuple[Qubit]:
        spacing = self.distance
        comp_qubits = tuple((spacing * i, 0, "computational") for i in range(m))
        anc_qubits = tuple(
            ((i - 0.5) * spacing, 0.5 * np.sqrt(3) * spacing, "ancilla")
            for i in range(m + 1)
        )

        return enumerate_qubits(comp_qubits + anc_qubits)


class DoubleTriangularLayoutAB(QubitLayout):
    def __init__(self, m: int, cutoff: float = 1, distance: float = 1) -> None:
        self.distance = distance
        super().__init__(m, cutoff)

    def __repr__(self):
        return f"Double triangular qubit layout ({self.m} comp. qubits, {self.n_ancilla} ancilla qubits)"

    def place_qubits(self, m: int) -> tuple[Qubit]:
        spacing = self.distance
        comp_qubits = tuple((spacing * i, 0, "computational") for i in range(m))
        anc_qubits_t = tuple(
            ((i - 0.5) * spacing, 0.5 * np.sqrt(3) * spacing, "ancilla")
            for i in range(m)
        )
        anc_qubits_l = tuple(
            ((i - 0.5) * spacing, -0.5 * np.sqrt(3) * spacing, "ancilla")
            for i in range(m)
        )

        return enumerate_qubits(comp_qubits + anc_qubits_t + anc_qubits_l)


class TriangularLayoutA(QubitLayout):
    def __init__(self, m: int, cutoff: float = 1, distance: float = 1) -> None:
        self.distance = distance
        super().__init__(m, cutoff)

    def __repr__(self):
        return f"Triangular qubit layout ({self.m} comp. qubits, {self.n_ancilla} ancilla qubits)"

    def place_qubits(self, m: int) -> tuple[Qubit]:
        spacing = self.distance
        comp_qubits_l = tuple((spacing * i, 0, "computational") for i in range(m))
        comp_qubits_t = tuple(
            ((i - 0.5) * spacing, 0.5 * np.sqrt(3) * spacing, "computational")
            for i in range(m)
        )

        return enumerate_qubits((comp_qubits_l + comp_qubits_t)[:m])


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


def HEA_fac(qubit_layout: QubitLayout, depth: int) -> Circuit:
    operations = [
        ("rz", "AB"),
        ("rx", "AB"),
        ("rz", "AB"),
        ("ryd ent", ""),
    ] * depth

    return unitary_circuit_fac(qubit_layout, operations)


def SHEA_trot_fac(
    qubit_layout: QubitLayout, H: np.ndarray, t: float, depth: int
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
        ("ryd ent", ""),
    ] * depth

    return unitary_circuit_fac(qubit_layout, operations)


def SHEA_fac(qubit_layout: QubitLayout, H: np.ndarray, t: float, depth: int) -> Circuit:
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
        ("ryd ent", ""),
    ] * depth

    return unitary_circuit_fac(qubit_layout, operations)
