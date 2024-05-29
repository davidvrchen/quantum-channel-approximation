from q_lab_toolbox.pprint.user import *

m = 1
omegas = (0.5,)
gammas = (0.5,)
ryd_interaction = 1.8

system = DecaySystem(ryd_interaction=ryd_interaction, omegas=omegas, m=m, gammas=gammas)

rho0s = random_rho0s(m=1, L=10)
rhoss, ts = solve_lindblad_rho0s(rho0s=rho0s, delta_t=0.5, N=3, s=system)
Os = all_observables(m=1)
training_data = mk_training_data(rhoss, Os)

rhoss_short, ts_short = solve_lindblad_rho0s(rho0s=rho0s, delta_t=0.1, N=3, s=system)
training_data_short = mk_training_data(rhoss_short, Os)
