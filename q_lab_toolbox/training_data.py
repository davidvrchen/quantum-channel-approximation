from dataclasses import dataclass

import numpy as np
import qutip as qt

from q_lab_toolbox.type_hints import (
    Observable,
    DensityMatrix,
    DensityMatrices,
    DensityMatricess,
)

from q_lab_toolbox.physics_defns.target_systems import TargetSystem
from q_lab_toolbox.physics_defns.hamiltonians import create_hamiltonian
from q_lab_toolbox.physics_defns.jump_operators import create_jump_operators
from q_lab_toolbox.physics_defns.initial_states import rho_rand_haar


@dataclass
class TrainingData:
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
        K_Os = len(self.Os)
        self.dims_A, _ = self.Os[0].shape
        self.L, K_Ess, self.N_ = self.Ess.shape
        self.N = self.N_ - 1

        assert (
            K_Os == K_Ess
        ), f"Number of observables {K_Os} does not match number of expecation values {K_Ess}"

        self.K = K_Os
        self.m = self.dims_A.bit_length() - 1


def random_rho0s(m: int, L: int, seed: int = None) -> DensityMatrices:

    if seed is None:
        seed = np.random.randint(10**5)
        print(f"random_rho0s: setting {seed=}")

    rng = np.random.default_rng(seed=seed)

    seeds = [rng.integers(0, 1000) for _ in range(L)]
    rho0s = [rho_rand_haar(m=m, seed=seed) for seed in seeds]

    return rho0s


def solve_lindblad_rho0(
    rho0: DensityMatrix, delta_t: float, N: int, s: TargetSystem
) -> tuple[DensityMatricess, np.ndarray]:

    H = create_hamiltonian(s)
    An = create_jump_operators(s)

    ts = np.arange(N + 1) * delta_t

    rhoss = qt.mesolve(H=H, rho0=rho0, tlist=ts, c_ops=An).states

    return rhoss, ts


def solve_lindblad_rho0s(
    rho0s: DensityMatrices, delta_t: float, N: int, s: TargetSystem
) -> tuple[DensityMatricess, np.ndarray]:

    H = create_hamiltonian(s)
    An = create_jump_operators(s)

    L = len(rho0s)
    dims, _ = rho0s[0].shape

    ts = np.arange(N + 1) * delta_t

    rhoss = np.zeros((L, N + 1, dims, dims), dtype=np.complex128)

    for l in range(L):
        rhoss[l, :, :, :] = qt.mesolve(H=H, rho0=rho0s[l], tlist=ts, c_ops=An).states

    return rhoss, ts


def measure_rhos(rhos: DensityMatrices, Os: list[Observable]) -> np.ndarray:
    return np.einsum("kab,nab -> kn", Os, rhos, dtype=np.float64, optimize="greedy")


def measure_rhoss(rhoss: np.ndarray, Os: list[Observable]) -> np.ndarray:

    return np.einsum("kab, lnba -> lkn", Os, rhoss, dtype=np.float64, optimize="greedy")


def mk_training_data(rhoss: DensityMatricess, Os: list[Observable]) -> TrainingData:

    rho0s = rhoss[:, 0, :, :]
    Ess = measure_rhoss(rhoss, Os)

    return TrainingData(Os, rho0s, Ess)
