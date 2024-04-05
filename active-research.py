from q_lab_toolbox.target_systems import DecaySystem
from q_lab_toolbox.hamiltonians import create_hamiltonian
from q_lab_toolbox.jump_operators import create_jump_operators
from q_lab_toolbox.initial_states import _rho_rand_haar
import numpy as np
from q_lab_toolbox.training_data import mk_training_data_states
from q_lab_toolbox.readout_operators import _order_n_observables
from q_lab_toolbox.channels import GateBasedChannel

from q_lab_toolbox.unitary_circuits import HardwareAnsatz

decay_system = DecaySystem(
    ryd_interaction=0.1, omegas=(0.2, 0.4), m=2, gammas=(0.3, 0.35)
)

H = create_hamiltonian(decay_system)
An = create_jump_operators(decay_system)


rho00 = _rho_rand_haar(m=2, seed=3)
rho10 = _rho_rand_haar(m=2, seed=4)

N = 3
ts = np.linspace(0, 10, N + 1)


rhoss = mk_training_data_states([rho00, rho10], ts, decay_system)


circuit = HardwareAnsatz(
    m=2, n_qubits=5, depth=5, gate_type="ryd", structure="triangle"
)

Os = _order_n_observables(m=2, n=1)

training_data = rhoss, Os

print( circuit.J_from_states(circuit.init_flat_theta(), training_data) )