import numpy as np
import qutip as qt

from q_lab_toolbox.target_systems import TargetSystem

from q_lab_toolbox.hamiltonians import create_hamiltonian
from q_lab_toolbox.jump_operators import create_jump_operators


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
    """Measured rhos with observables Os
    and returns the expectation values as a matrix.

    Args:
    ----
        rhos (list[qt.Qobj]): states at different times

        Os (list[qt.Qobj]): set of observables

    Returns:
    -------
        Ess (np.ndarray): matrix of the expectation values
        using the indexing convention as defined in the paper
        it is indexed as Ess_k,n

    First make the cartesian product of Os and rhos
    then take multiply and take the trace
    We will get something like:\n
    Tr[Os[0] rhos[0]]  Tr[Os[0] rhos[1]] ......... Tr[Os[0] rhos[N]]\n
    Tr[Os[1] rhos[0]]  Tr[Os[1] rhos[1]] ......... Tr[Os[1] rhos[N]]\n
        \..........\n
    Tr[Os[L-1] rho[0]] Tr[Os[L-1] rhos[1]] ... Tr[Os[L-1] rhos[N]]
    """

    _N = len(rhos)
    K = len(Os)

    dims = Os[0].dims

    Ess = np.zeros((K, _N), dtype=np.float64)

    for k, O in enumerate(Os):
        Ess[k, :] = np.array([(O * qt.Qobj(rho, dims=dims)).tr() for rho in rhos])

    return Ess

def measure_rhoss(rhoss: np.ndarray, Os: list[qt.Qobj]) -> np.ndarray:
    L, _N, _, _ = rhoss.shape
    K = len(Os)
    Ess = np.zeros((L, K, _N), dtype=np.float64)

    for l, rhos in enumerate(rhoss):
        Ess[l, :, :] = measure_rhos(rhos, Os)