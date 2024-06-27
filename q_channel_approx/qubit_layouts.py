import itertools
from abc import ABC, abstractmethod

from typing import Callable, Iterable, NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


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

    def show_layout(self, title: bool =True, c_map: dict = None) -> Axes:
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

        plt.legend(handles=patches, bbox_to_anchor=(0.8, 0.6))
        if title:
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
        r = m // 2
        comp_qubits_l = tuple((spacing * i, 0, "computational") for i in range(r))
        comp_qubits_t = tuple(
            ((i - 0.5) * spacing, 0.5 * np.sqrt(3) * spacing, "computational")
            for i in range(r+1)
        )

        return enumerate_qubits((comp_qubits_l + comp_qubits_t)[:m])
