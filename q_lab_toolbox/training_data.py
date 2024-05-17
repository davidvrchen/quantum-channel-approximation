from dataclasses import dataclass, KW_ONLY
import random

import numpy as np
import qutip as qt

from q_lab_toolbox.target_systems import TargetSystem

from q_lab_toolbox.hamiltonians import create_hamiltonian
from q_lab_toolbox.jump_operators import create_jump_operators

from q_lab_toolbox.readout_operators import Observables, create_observables
from q_lab_toolbox.initial_states import RhoRandHaar, create_rho0


@dataclass
class _TrainingData:
    """Training data is defined as a list of observales Os together with
    a list of initial states rho0s and a grid of expectation values Ess.
    The class automatically extracts some handy variables such as the dimensions of the
    underlying Hilbert space `dims`, and the indexing variables `K, L, N`.

    Args:
    -----
        Os (np.ndarray): "list of observables", but an observables
        is a matrix so this should be a 3D array indexed by `(k, a, b)`
        where `k` indexes the observable, and `a` and `b` are the row and column
        index of the matrix respectively.
            `[O_0, O_1, O_2, ..., O_K]`

        rho0s (np.ndarray): "matrix of states" each row gives the
        evolution of a particular initial state, but since a state is a density matrix
        this is a 4D array indexed by `(l, n, a, b)` where `l` indexes the initial state
        `n` indexes the time step and `a` and `b` respectively index the row and column
        of the density matrix.
            `       N ->`\n
            `L   [[rho00, rho01, ..., rho0N],`\n
            `|    [rho10, rho11, ..., rho1N],`\n
            `v     ...`\n
                 `[rhoL0, rhoL1, ..., rhoLN]]`

        Ess (np.ndarray): "list of expectation values of each states with each observable"
        but since there are `L` initial states and `K` observables it is a list of matrices
        or a 3D array. The indexing convention is (l, k, n).
    """

    Os: np.ndarray
    rho0s: np.ndarray
    Ess: np.ndarray

    def __post_init__(self):
        """Determine the indexing variables `K, L, N`,
        the dimension of the underlying Hilbert space.
        """
        self.K, dims_O, _ = self.Os.shape
        self.L, self.N, dims_rho, _ = self.rhos.shape

        # check if dimensions of Hilbert spaces according to
        # states and observables match
        assert (
            dims_O == dims_rho
        ), f"Dimensions of observables {dims_O} and states {dims_rho} do not match"

        self.dims_A = dims_O

@dataclass
class TrainingData:
    target_system: TargetSystem
    N: int
    delta_t: float
    Os: Observables
    rho0s: list[qt.Qobj] = None

@dataclass
class RandomTrainingData(TrainingData):

    _: KW_ONLY
    seed: int
    L: int

    def __post_init__(self):
        random.seed(self.seed)
        seeds = [random.randint(0, 1000) for _ in range(self.L)]
        self.rho0s = [
            create_rho0(RhoRandHaar(self.target_system.m, seed)) for seed in seeds
        ]


def solve_lindblad(rho0: qt.Qobj, ts: np.ndarray, s: TargetSystem):
    """Solves the Lindblad equation with initial state rho0 on times ts,

    Args:
    -----
    rho0 (qt.Qobj): the initial state

    ts (np.ndarray): the times at which to integrate the states

    s (TargetSystem): the target system

    Returns:
    -------
        rhos, where rhos[i] is the state at time ts[i]
    """

    H = create_hamiltonian(s)
    An = create_jump_operators(s)

    result = qt.mesolve(H=H, rho0=rho0, tlist=ts, c_ops=An)

    rhos = result.states

    return rhos


def mk_training_data_states(rho0s, ts, s):

    _N = len(ts)
    L = len(rho0s)

    m = s.m

    rhoss = np.zeros((L, _N, 2**m, 2**m), dtype=np.complex128)
    for l, rho0 in enumerate(rho0s):
        rhoss[l, :, :, :] = solve_lindblad(rho0, ts, s)

    return rhoss


def measure_rhos(rhos: list[qt.Qobj], Os: list[qt.Qobj]) -> np.ndarray:
    Ess = np.einsum("kab,nab -> kn", Os, rhos)

    return Ess


def measure_rhoss(rhoss: np.ndarray, Os: list[qt.Qobj]) -> np.ndarray:

    Ess = np.einsum("kab, lnba -> lkn", Os, rhoss, dtype=np.float64)

    return Ess


def mk_training_data(s: TrainingData):

    ts = np.arange(s.N + 1) * s.delta_t

    Os = np.array(create_observables(s.Os))

    rhoss = mk_training_data_states(s.rho0s, ts, s.target_system)

    Ess = np.einsum("kab, lnba -> lkn", Os, rhoss)

    return ts, Ess, Os
