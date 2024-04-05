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

    # the -1 is added to match the definition of N as per the report
    N = len(rhos) - 1
    K = len(Os)

    Ess = np.zeros((K, N + 1), dtype=np.float64)

    for k, O in enumerate(Os):
        Ess[k, :] = np.array([(O * rho).tr() for rho in rhos])

    return Ess
