import numpy as np
import matplotlib.pyplot as plt

from q_lab_toolbox.observables import (
    create_readout_computational_basis,
    computation_basis_labels,
    all_observables,
)

from q_lab_toolbox.channel import channel_fac, evolver_fac
from q_lab_toolbox.optimizer import optimize
from q_lab_toolbox.physics_defns.initial_states import rho_rand_haar
from q_lab_toolbox.pprint.visualize import plot_ess, compare_ess
from q_lab_toolbox.unitary_circuits import (
    _HEA_fac,
    TriangularLayout,
    unitary_circuit_fac,
)
from q_lab_toolbox.physics_defns.target_systems import DecaySystem
from q_lab_toolbox.training_data import (
    random_rho0s,
    solve_lindblad_rho0s,
    mk_training_data,
    measure_rhos,
    solve_lindblad_rho0,
)


m = 1
omegas = (0.5,)
gammas = (0.5,)
ryd_interaction = 1.8

system = DecaySystem(ryd_interaction=1.8, omegas=(0.5,), m=1, gammas=(0.5,))
