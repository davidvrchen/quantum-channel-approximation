import matplotlib.pyplot as plt
import numpy as np
from q_channel_approx.physics_defns import (
    DecaySystem,
    rho_rand_haar,
    default_jump_operators
)
from q_channel_approx.training_data import (
    random_rho0s,
    solve_lindblad_rho0s,
    mk_training_data,
    measure_rhos,
    solve_lindblad_rho0
)
from q_channel_approx.training_observables import all_observables
from q_channel_approx.unitary_circuits import TriangularLayoutA, TriangularLayoutAB, unitary_circuit_fac
from q_channel_approx.channel import evolver_fac
from q_channel_approx.pprint import comp_basis_labels
from q_channel_approx.plotting.plot_observables import create_observables_comp_basis
from q_channel_approx.optimizer import optimize
from q_channel_approx.plotting.visualize import compare_ess

m = 1
omegas = (0.4,)
gammas = (0.3,)
ryd_interaction = 0.2

jump_oper = default_jump_operators(m, gammas)
system = DecaySystem(ryd_interaction=ryd_interaction, omegas=omegas, m=m, gammas=gammas)

rho0s = random_rho0s(m=1, L=10)
rhoss, ts = solve_lindblad_rho0s(rho0s=rho0s, delta_t=0.5, N=1, s=system, jump_opers=jump_oper)
Os = all_observables(m=1)
training_data = mk_training_data(rhoss, Os)

rhoss_short, ts_short = solve_lindblad_rho0s(rho0s=rho0s, delta_t=0.1, N=1, s=system, jump_opers=[])
training_data_short = mk_training_data(rhoss_short, Os)
