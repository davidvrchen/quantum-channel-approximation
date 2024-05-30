from dataclasses import dataclass

import numpy as np
import qutip as qt

from q_channel_approx.physics_defns.target_systems import TargetSystem
from q_channel_approx.physics_defns.hamiltonians import create_hamiltonian
from q_channel_approx.physics_defns.initial_states import rho_rand_haar


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


def random_rho0s(m: int, L: int, seed: int = None) -> list[qt.Qobj]:
    """Generate a list of `L` initial states on `m` qubits.

    Args:
        m (int): number of qubits.
        L (int): number of initial states.
        seed (int, optional): used for the generation of L seed values
        which are passed to `rho_rand_haar`. Defaults to None.

    Returns:
        list[qt.Qobj]: list of `L` randomly generated initial states.
    """

    if seed is None:
        seed = np.random.randint(10**5)
        print(f"random_rho0s: setting {seed=}")

    rng = np.random.default_rng(seed=seed)

    seeds = [rng.integers(0, 10**5) for _ in range(L)]
    rho0s = [rho_rand_haar(m=m, seed=seed) for seed in seeds]

    return rho0s


def solve_lindblad_rho0(
    rho0: qt.Qobj,
    delta_t: float,
    N: int,
    s: TargetSystem,
    jump_opers: list[qt.Qobj],
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve a single initial state `rho0` for `N` timesteps of `delta_t` according the
    Lindblad equation with Hamiltonian defined by `s` and using
    jump operators `jump_opers`

    Args:
        rho0 (qt.Qobj): initial state
        delta_t (float): time step
        N (int): number of time steps
        s (TargetSystem): settings object used to create Hamiltonian
        jump_opers (list[qt.Qobj]): list of jump operators

    Returns:
        tuple[list[qt.Qobj], np.ndarray]: evolution of the initial state,
        list of timesteps at which the states are given
    """

    H = create_hamiltonian(s)

    ts = np.arange(N + 1) * delta_t

    rhoss = qt.mesolve(H=H, rho0=rho0, tlist=ts, c_ops=jump_opers).states

    return rhoss, ts


def solve_lindblad_rho0s(
    rho0s: list[qt.Qobj],
    delta_t: float,
    N: int,
    s: TargetSystem,
    jump_opers: list[qt.Qobj],
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve all `rho0s` for `N` timesteps of `delta_t` according the
    Lindblad equation with Hamiltonian defined by `s` and using
    jump operators `jump_opers`

    Args:
        rho0s (list[qt.Qobj]): list of initial states
        delta_t (float): time step between states
        N (int): Number of evolutions of delta_t to make
        s (TargetSystem): settings object used to create the Hamiltonian
        jump_opers (list[qt.Qobj]): jump operators for the Lindbladian

    Returns:
        tuple[np.ndarray, np.ndarray]: L x N matrix
        where each entry is a density matrix itself, ts
        which is a list of time steps at which the states are given.

    """

    H = create_hamiltonian(s)

    L = len(rho0s)
    dims, _ = rho0s[0].shape

    ts = np.arange(N + 1) * delta_t

    rhoss = np.zeros((L, N + 1, dims, dims), dtype=np.complex128)

    for l in range(L):
        rhoss[l, :, :, :] = np.array(
            [
                state.full()
                for state in qt.mesolve(
                    H=H, rho0=rho0s[l], tlist=ts, c_ops=jump_opers
                ).states
            ]
        )

    return rhoss, ts


def measure_rhos(rhos: np.ndarray, Os: list[np.ndarray]) -> np.ndarray:
    """Create a matrix of expectation values by measuring (i.e. trace of O rho)
    a list of density matrices with a list of observables.
    If there are `K` observables in `Os` and `N` states in `rhos`
    then the resulting matrix is of dimension `K` by `N`.

    Args:
        rhos (np.ndarray): think of it as a list of density matrices (length `N`).
        Os (list[np.ndarray]): think of it as a list of observables (length `K`).

    Returns:
        np.ndarray: matrix of expectation values of dimension `K` by `N`.
    """
    return np.einsum("kab,nab -> kn", Os, rhos, dtype=np.float64, optimize="greedy")


def measure_rhoss(rhoss: np.ndarray, Os: list[np.ndarray]) -> np.ndarray:
    """Create a holor of expectation values by measuring (i.e. trace of O rho)
    a matrix of density matrices with a list of observables.
    If there are `K` observables in `Os` and `rhoss` is of dimension (`L`, `N`)
    then the resulting matrix is of dimension `K` by `N`.

    Args:
        rhoss (np.ndarray): think of it as a list of density matrices (dims `L` by `N`).
        Os (list[np.ndarray]): think of it as a list of observables (length `K`).

    Returns:
        np.ndarray: holor of expectation values (dimension (`L`, `K`, `N`)).
    """
    return np.einsum("kab, lnba -> lkn", Os, rhoss, dtype=np.float64, optimize="greedy")


def mk_training_data(rhoss: np.ndarray, Os: list[qt.Qobj]) -> TrainingData:
    """Create training data object from a matrix of states where each row
    gives the evolution of its zeroth state and a list of observables.

    Args:
        rhoss (np.ndarray): matrix of states
        Os (list[qt.Qobj]): list of observables

    Returns:
        TrainingData: the corresponding TrainingData object
        which can be used to optimize a gate sequence.
    """

    rho0s = rhoss[:, 0, :, :]
    Os = [O.full() for O in Os]
    Ess = measure_rhoss(rhoss, Os)

    return TrainingData(Os, rho0s, Ess)
