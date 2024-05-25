from abc import ABC, abstractmethod
from typing import NamedTuple, Iterable
import itertools
import functools

import scipy as sc
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numba import jit, prange

from q_lab_toolbox.type_hints import Unitary, UnitaryFactory, Theta, Hamiltonian


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


class QubitLayout:
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
        return f"""generic qubit layout"""

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

        # TO DO :: Implement better cutoff check, i.e. less than a
        # threshold value which is set when creating the qubit layout
        # but first consult Luke / Robert on units used in the program
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

        if c_map is None:
            c_map = {"computational": "#1f77b4", "ancilla": "#ff7f0e"}

        xs_comp, ys_comp, _, id_comp = zip(*self.comp_qubits)
        xs_anc, ys_anc, _, id_anc = zip(*self.anc_qubits)

        plt.scatter(
            xs_anc,
            ys_anc,
            c=c_map["ancilla"],
            s=500,
            label="Ancilla",
            edgecolors="none",
        )
        plt.scatter(
            xs_comp,
            ys_comp,
            c=c_map["computational"],
            s=500,
            label="Computational",
            edgecolors="none",
        )

        ax = plt.gca()

        OFFSET_X = -0.25
        OFFSET_Y = -0.2
        for i, id in enumerate(id_anc + id_comp):
            ax.annotate(
                id,
                ((xs_anc + xs_comp)[i] + OFFSET_X, (ys_anc + ys_comp)[i] + OFFSET_Y),
                fontsize=8,
                weight="bold",
            )

        ax.set_aspect("equal")
        ax.margins(x=0.3, y=0.8)
        plt.axis("off")

        lgnd = plt.legend(loc="lower right", scatterpoints=1, fontsize=10)

        lgnd.legend_handles[0]._sizes = [30]
        lgnd.legend_handles[1]._sizes = [30]

        plt.title("Qubit layout", weight="bold")
        return ax


class HalfTriangularLayout(QubitLayout):
    def __repr__(self) -> str:
        return """half triangular qubit layout"""

    def place_qubits(self, m) -> tuple[Qubit]:
        spacing = 1
        comp_qubits = tuple((spacing * i, 0, "computational") for i in range(m))
        anc_qubits = tuple(
            ((i - 0.5) * spacing, 0.5 * np.sqrt(3), "ancilla") for i in range(m + 1)
        )

        return enumerate_qubits(comp_qubits + anc_qubits)


class FullTriangularLayout(QubitLayout):
    def __repr__(self) -> str:
        return """full triangular qubit layout"""

    def place_qubits(self, m) -> tuple[Qubit]:
        spacing = 1
        comp_qubits = tuple((spacing * i, 0, "computational") for i in range(m))
        anc_qubits_t = tuple(
            ((i - 0.5) * spacing, 0.5 * np.sqrt(3), "ancilla") for i in range(m)
        )
        anc_qubits_l = tuple(
            ((i - 0.5) * spacing, -0.5 * np.sqrt(3), "ancilla") for i in range(m)
        )

        return enumerate_qubits(comp_qubits + anc_qubits_t + anc_qubits_l)


def kron_gates_l(single_gates):
    result = single_gates[0]
    for gate in single_gates[1:]:
        result = np.kron(result, gate)

    return result


def kron_neighbours_even(single_gates):

    l, dims, _ = single_gates.shape
    double_gates = np.zeros((l // 2, dims**2, dims**2), dtype=np.complex128)

    for i in prange(0, l // 2):
        double_gates[i, :, :] = np.kron(single_gates[i * 2], single_gates[i * 2 + 1])

    return double_gates


def kron_gates_t(single_gates):
    """Recursively multiply the neighbouring gates.
    When the block size gets below the turnover point the linear
    kron_gates_l is used as it is more efficient in this usecase."""
    TURNOVER = 3

    l = len(single_gates)

    if l > TURNOVER:
        if l % 2 == 0:
            return kron_gates_t(kron_neighbours_even(single_gates))
        return np.kron(
            kron_gates_t(kron_neighbours_even(single_gates[:-1, :, :])),
            single_gates[-1],
        )

    return kron_gates_l(np.array(single_gates))


def dims2qubit(dims: int) -> int:
    """Calculate the amount of qubits needed to make a Hilbert space of dimension dims.

    Args:
        dims (int): the dimension of the Hilbert space.

    Returns:
        int: the number of qubits that would be needed.
    """

    return dims.bit_length() - 1


class GateOperation(NamedTuple):
    """Gate operation

    Args:
        target_system ("A", "B", "AB"): System to actively operate on.
        gate_type ("rx "ry", "rz", "ryd", "xy", "cnot"): The gate function to use on the target system
    """

    target_sytem: str
    gate_type: str


def rz(theta):
    zero = np.zeros(theta.shape)
    exp_m_theta = np.exp(-1j * theta / 2)
    exp_theta = np.exp(1j * theta / 2)

    single_gates = np.einsum(
        "ijk->kij", np.array([[exp_m_theta, zero], [zero, exp_theta]])
    )

    u_gates = kron_gates_t(single_gates)

    return u_gates


def U_fac(qubit_layout: QubitLayout, gate_operations: [GateOperation], ):

    dims_dict = {
        "A": qubit_layout.dims_A,
        "B": qubit_layout.dims_B,
        "AB": qubit_layout.dims_AB,
    }

    gate_operations = [
        (dims_dict[gate.target_system], gate.gate_type) for gate in gate_operations
    ]

    for gate in gate_operations:
        target, gate_type = gate
        match gate_type:
            case "rx":
                rx(targetdims, qubit_layout.dims_AB)

            case "rz":
                rz(target)


class GateFunction(ABC):

    def __repr__(self) -> str:
        return "Unnamed gate operation"

    @abstractmethod
    def u_fac(
        self, dims_target: int, dims_AB: int, connections: tuple[GateConnection]
    ) -> tuple[UnitaryFactory, int]:
        """Create (u, p) where u is a unitary factory and p is the number of
        parameters it expects.

        Args:
            dims_target (int): the dimension to which to apply the gate operation.
            dims_AB (int): the dimension of u(theta).

        Returns:
            tuple[UnitaryFactory, int]: pair of a unitary factory and the number of parameters it expects.
        """
        pass


class HamiltonianA(GateFunction):

    def __init__(self, H: Hamiltonian, t: float) -> None:

        if isinstance(H, qt.Qobj):
            self.H = H.full()
        else:
            self.H = H

        self.t = t

        self.dims, _ = self.H.shape

    def __repr__(self) -> str:
        return "Hamiltonian"

    def u_fac(
        self, dims_target: int, dims_AB: int, connections: tuple[GateConnection]
    ) -> Unitary:

        assert (
            dims_target == self.dims
        ), f"Dimensions do not match: {dims_target=} != {self.dims=}"

        dims_expand = dims_AB - dims_target

        e_H = sc.linalg.expm((-1j) * self.t * self.H)
        e_H_exp = np.kron(e_H, np.identity(dims_expand))

        @jit(forceobj=True)
        def u(theta):
            return e_H_exp

        return u, 0


class RX(GateFunction):

    def __repr__(self) -> str:
        return "RX gates"

    def u_fac(
        self, dims_target, dims_AB, connections: tuple[GateConnection]
    ) -> UnitaryFactory:

        dims_expand = dims_AB - dims_target

        @jit(forceobj=True)
        def u(theta):

            costheta = np.cos(theta / 2)
            sintheta = np.sin(theta / 2)

            single_gates = np.einsum(
                "ijk->kij",
                np.array(
                    [
                        [costheta, -sintheta],
                        [sintheta, costheta],
                    ]
                ),
            )

            u_gates = kron_gates_t(single_gates)

            return np.kron(u_gates, np.identity(dims_expand + 1))

        return u, dims2qubit(dims_target)


class RydEnt(GateFunction):

    def __repr__(self) -> str:
        return """rydberg entanglement"""

    def u_fac(
        self, dims_target: int, dims_AB: int, connections: tuple[GateConnection]
    ) -> tuple[UnitaryFactory, int]:

        assert (
            dims_target == dims_AB
        ), f"Dimensions of target and full system must math, {dims_target=}!={dims_AB=}"

        n_qubits = dims2qubit(dims_target)

        rydberg = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])

        def u(theta):
            rydberg_2gate = qt.Qobj(rydberg, dims=[[2] * 2, [2] * 2])
            rydberg_gate = np.zeros([dims_target, dims_target], dtype=np.complex128)
            for connection in connections:

                id1, id2, d = connection

                ham = qt.qip.operations.gates.expand_operator(
                    rydberg_2gate, n_qubits, [id1, id2]
                ).full()
                rydberg_gate += ham / d**3  # distance to the power -6

            return sc.linalg.expm(-1j * theta * rydberg_gate)

        return u, 1


class GateOperation(NamedTuple):
    """A gate operation on the qubits array, 2 fields.

    Args:
        system ("A", "B", "AB"): determines which
        Hilbert space the operation acts on.
            "A": Computational qubits
            "B": Ancilla qubits
            "AB": All qubits
        gate_f (GateFunction): The gate to be performed
    """

    system: str
    gate_f: GateFunction


def matmult_l(us):
    result = us[0]
    for u in us[1:]:
        result = result @ u

    return result


class GateBasedUnitaryCircuit(ABC):
    def __init__(
        self,
        qubit_layout: QubitLayout,
        operations: tuple[GateOperation],
        D: int = 1,
        repeats: int = 1,
    ):
        self.qubit_layout = qubit_layout
        self.operations = operations

        self.n = len(operations)

        self.D = D
        self.repeats = repeats

    def __repr__(self) -> str:
        return f"""Gate based unitary circuit on {self.qubit_layout} \r
Gate operations: \r {[operation.gate_f for operation in self.operations]} \r
repeats: {self.repeats}, D: {self.D}"""

    def U_fac(self) -> UnitaryFactory:
        """Returns function to compute U operator (as matrix) that represents the unitary circuit
        with parameters theta."""

        us = np.zeros(
            (self.n * self.D, self.qubit_layout.dims_AB, self.qubit_layout.dims_AB),
            dtype=np.complex128,
        )
        from operator import add

        ps = [pair[1] for pair in self.up_pairs]
        par_inds = [0] + list(itertools.accumulate(ps, add))

        n = self.n
        repeats = self.repeats
        D = self.D

        up_pairs = self.up_pairs
        u = [up[0] for up in up_pairs]

        single = up_pairs[-1][1]

        @jit(parallel=True, forceobj=True)
        def U(theta: Theta) -> Unitary:
            for d in prange(D):
                for i in prange(n):
                    us[i, :, :] = u[i](
                        theta[d * single + par_inds[i] : d * single + par_inds[i + 1]]
                    )

            return np.linalg.matrix_power(matmult_l(us), repeats)

        return U

    @functools.cached_property
    def up_pairs(self):
        up_pairs = []
        for op in self.operations:

            match op.system:
                case "A":
                    dims = self.qubit_layout.dims_A
                case "B":
                    dims = self.qubit_layout.dims_B
                case "AB":
                    dims = self.qubit_layout.dims_AB

            up_pairs.append(
                op.gate_f.u_fac(
                    dims,
                    self.qubit_layout.dims_AB,
                    self.qubit_layout.gate_connections,
                )
            )

        return up_pairs

    def init_theta(self) -> Theta:
        """Creates array of zeroes with the right shape,
        i.e. such that U can work with the return."""

        ps = [pair[1] for pair in self.up_pairs]

        return np.ones(sum(ps) * self.D)
