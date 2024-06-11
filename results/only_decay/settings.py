import matplotlib.pyplot as plt
import numpy as np

from q_channel_approx.physics_defns import (
    NothingSystem,
    no_decay_jump_operators,
    default_jump_operators,
    create_hamiltonian,
    rho_rand_haar,
)
from q_channel_approx.training_data import (
    random_rho0s,
    solve_lindblad_rho0s,
    solve_lindblad_rho0,
    mk_training_data,
    measure_rhos,
)
from q_channel_approx.unitary_circuits import (
    TriangularLayoutA,
    TriangularLayoutAB,
    unitary_circuit_fac,
)
from q_channel_approx.training_observables import all_observables
from q_channel_approx.channel import evolver_fac
from q_channel_approx.plotting.observables import create_observables_comp_basis
from q_channel_approx.plotting.routines import compare_ess
from q_channel_approx.pprint import comp_basis_labels
from q_channel_approx.optimizer import optimize

m = 1
omegas = (0.5,)
gammas = (0.1,)
ryd_interaction = 1.8

system = NothingSystem(m=m, gammas=gammas)

rho0s = random_rho0s(m=1, L=10)

jump_oper_no_decay = no_decay_jump_operators()
jump_oper = default_jump_operators(m, gammas=gammas)

rhoss, ts = solve_lindblad_rho0s(rho0s=rho0s, delta_t=0.5, N=3, s=system, jump_opers=jump_oper)
Os = all_observables(m=1)
training_data = mk_training_data(rhoss, Os)

rhoss_short, ts_short = solve_lindblad_rho0s(rho0s=rho0s, delta_t=0.01, N=3, s=system, jump_opers=jump_oper)
training_data_short = mk_training_data(rhoss_short, Os)
