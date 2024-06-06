import numpy as np
from q_channel_approx.unitary_circuits import _HEA_fac, TriangularLayout
from q_channel_approx.physics_defns.target_systems import DecaySystem
from q_channel_approx.optimizer import optimize
from q_channel_approx.training_observables import all_observables
from q_channel_approx.training_data import (
    random_rho0s,
    solve_lindblad_rho0s,
    mk_training_data,
    solve_lindblad_rho0,
)
import matplotlib.pyplot as plt
from q_channel_approx.optimizer import channel_fac
from q_channel_approx.physics_defns.initial_states import rho_rand_haar
from q_channel_approx.plotting.plotting_routines import plot_ess, compare_ess
from q_channel_approx.training_data import measure_rhos

from q_channel_approx.training_observables import create_readout_computational_basis

from q_channel_approx.training_data import solve_lindblad_rho0s


qubits = TriangularLayout(m=1, cutoff=1, distance=0.9)
circuit = _HEA_fac(qubits, depth=3, repeats=5)


qubits.show_layout()


system = DecaySystem(ryd_interaction=0.2, omegas=(0.5,), m=1, gammas=(0.5,))

rho0s = random_rho0s(m=1, L=1000, seed=1)
rhoss, ts = solve_lindblad_rho0s(rho0s=rho0s, delta_t=0.5, N=2, s=system)
Os = all_observables(m=1)
training_data = mk_training_data(rhoss, Os)


theta_opt, errors, thetas = optimize(circuit, training_data, max_count=100)


rho0 = rho_rand_haar(1, 4)


def evolve_n_times(n: int, rho):
    rho_acc = rho
    rhos = [rho_acc]
    phi = channel_fac(circuit)(theta=theta_opt)
    for i in range(n):
        rho_acc = phi(rho=rho_acc)
        rhos.append(rho_acc)

    return np.array(rhos)


rhos = evolve_n_times(20, rho0)
Os = create_readout_computational_basis(1)
ess = measure_rhos(rhos, Os)


rho_ref_s, ts = solve_lindblad_rho0(rho0, delta_t=0.5, N=20, s=system)
e_ref_ss = measure_rhos(rho_ref_s, Os)


compare_ess((ts, ess, "approx"), (ts, e_ref_ss, "ref"), labels=["1", "2", "3", "4"])
