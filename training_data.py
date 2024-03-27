import numpy as np
import qutip as qt
import scipy as sc

from dataclasses import dataclass, KW_ONLY
from q_lab_toolbox.utils.my_functions import get_paulis
from q_lab_toolbox.settings import TargetSystemSettings, DecaySettings

from q_lab_toolbox.jump_operators import create_jump_operators
from q_lab_toolbox.hamiltonians import create_hamiltonian


@dataclass
class TrainingDataSettings:

    m: int
    n_training: int
    seed: int
    paulis: str
    t_repeats: int
    n_measurements: int


@dataclass
class TrainingData:
    """Dataclass that represents training data.

    Parameters:
    ----------

    rho0: the initial state

    n: number of time steps

    rhos: 
    """
    _: KW_ONLY

    rho0: qt.Qobj
    n: int = 1
    rhos: list[qt.Qobj]
    Os: list[qt.Qobj]


def lviss_solver(H: qt.Qobj, An: qt.Qobj, t_H: float) -> qt.Qobj:

    dim = H.shape[0]

    def lindblad_evolution(t_eval, rho0):

        def lindbladian(t, rho):
            rho = np.reshape(rho, (dim, dim))
            result = -1j * (H * rho - rho * H)
            for A in An:
                Ad = np.conj(np.transpose(A))
                result = result + A * rho * Ad - Ad * A * rho / 2 - rho * Ad * A / 2
            result = np.reshape(result, 2**dim)
            return result

        solver = sc.integrate.complex_ode(lindbladian)
        solver.set_initial_value(np.reshape(rho0, 2**dim), 0)

        if type(t_eval) == np.ndarray:
            sol = np.zeros([len(t_eval), 2**dim], dtype=np.complex128)
            sol[0] = solver.integrate(t_eval[0])
            for i in range(1, len(t_eval)):
                solver.set_initial_value(sol[i - 1], t_eval[i - 1])
                sol[i] = solver.integrate(t_eval[i])
            return np.reshape(sol, [len(t_eval), dim, dim])

        else:
            sol = solver.integrate(t_eval)
            if solver.successful():
                return np.reshape(sol, [dim, dim])
            else:
                print("Solver for lindblad evolution aborted")
                return rho0

    evolution = lambda rho0: lindblad_evolution(t_H, rho0)

    return evolution


def set_steady_state(evolution, m):
    """ """
    random_ket = qt.rand_ket_haar(dims=[[2**m], [1]])
    random_ket.dims = [[2] * m, [2] * m]
    random_bra = random_ket.dag()
    steady_state_old = (random_ket * random_bra).full()
    steady_state_new = evolution(steady_state_old)
    count = 0
    maxcount = 5000
    while (
        np.amax(np.abs(steady_state_old - steady_state_new)) > 10 ** (-6)
        and count < maxcount
    ):
        steady_state_old = steady_state_new
        steady_state_new = evolution(steady_state_old)
        count += 1
    if count == maxcount:
        print("Steady state not found")
    steady_state = steady_state_new
    return steady_state


def evolution_n(evolution, n, rho: qt.Qobj):
    """
    Calculate the exact evolution for n time steps of rho

    Parameters
    ----------
    n : int
        number of time steps.
    rho : np.array, 2**m x 2**m
        density matrix.

    Returns
    -------
    rho_end : np.array, n+1 x 2**m x 2**m
        list of evolutions of rho.

    """

    # dimension of Hilbert space
    dim = rho.shape[0]

    rho_end = np.zeros((n + 1, dim, dim), dtype=np.complex128)
    rho_end[0] = rho
    for i in range(n):
        rho = evolution(rho)
        rho_end[i + 1] = rho
    return rho_end


def mk_training_data(s_data: TrainingDataSettings, s_target: TargetSystemSettings):
    """
    TrainingDataSettings -> training_data

    """

    # read parameters from settings related what kind of
    # training data to make
    m = s_data.m
    seed = s_data.seed
    n_training = s_data.n_training
    paulis = s_data.paulis
    t_repeats = s_data.t_repeats
    n_measurements = s_data.n_measurements

    # read parameters from settings related to the target system
    H = create_hamiltonian(s=s_target)
    An = create_jump_operators(s=s_target)
    t_H = 0.1

    evolution = lviss_solver(H, An, t_H)

    training = np.zeros((t_repeats + 1, n_training, 2**m, 2**m), dtype=np.csingle)

    for l in range(n_training):
        if l == 0:
            random_ket = qt.rand_ket_haar(dims=[[2**m], [1]], seed=seed)
            random_ket.dims = [[2] * m, [2] * m]
            random_bra = random_ket.dag()
            rho = random_ket * random_bra
            np.random.seed(seed)
        elif l == n_training - 1:
            steady_state = set_steady_state(evolution=evolution, m=m)
            rho = steady_state
        else:
            mix_factor = np.random.rand() ** 1 / 2

            evals = np.random.normal(size=2**m)
            evals = evals**2 / np.sum(evals**2)

            zero_mat = np.zeros((2**m, 2**m))
            zero_mat[0, 0] = 1

            init_matrix = mix_factor * zero_mat + (1 - mix_factor) * np.diag(evals)
            random_mixed = qt.Qobj(init_matrix, dims=[[2] * m, [2] * m])

            U = qt.random_objects.rand_unitary_haar(N=2**m, dims=[[2] * m, [2] * m])
            rho = U * random_mixed * U.dag()

        training[:, l, :, :] = np.reshape(
            evolution_n(evolution, t_repeats, rho), (t_repeats + 1, 2**m, 2**m)
        )

    # self.training_rho = rho_list
    training_data = training

    return training_data


from q_lab_toolbox.settings import TargetSystemSettings
from q_lab_toolbox.hamiltonians import create_hamiltonian
from q_lab_toolbox.jump_operators import create_jump_operators
from q_lab_toolbox.qubit_readout_operators import create_readout_computational_basis
from q_lab_toolbox.initial_states import rho_rand_haar
from q_lab_toolbox.initial_states import RandHaarSettings

def mk_training_data2(rho0, delta_t, n_training, Os, s: TargetSystemSettings):

    H = create_hamiltonian(s)
    An = create_jump_operators(s)

    ts = np.arange(n_training) * delta_t

    result = qt.mesolve(H=H, rho0=rho0, tlist=ts, c_ops=An)

    rhos = result.states
    L = len(Os)
    Ess = np.zeros((L, n_training), dtype=np.float64)

    # print(rhos)
    # print(result)

    for l, O in enumerate(Os):
        Ess[l, :] = np.array( [ (O * rho).tr() for rho in rhos] )


    return rho0, Os, Ess

if __name__ == "__main__":
    s_data = TrainingDataSettings(
        m=2, n_training=10, seed=5, paulis="order 1", t_repeats=2, n_measurements=3
    )
    s_target = DecaySettings(
        m=2, gammas=(0.3, 0.5), ryd_interaction=0.1, omegas=(0.2, 0.4)
    )

    rho0_s = RandHaarSettings(m=2, seed=5)
    rho0 = rho_rand_haar(rho0_s)

    Os = create_readout_computational_basis(s_target)

    data = mk_training_data2(rho0, 0.1, 3, Os, s_target)
    print(data)