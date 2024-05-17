from q_lab_toolbox.target_systems import DecaySystem
from q_lab_toolbox.hamiltonians import create_hamiltonian
from q_lab_toolbox.jump_operators import create_jump_operators
from q_lab_toolbox.initial_states import _rho_rand_haar
import numpy as np
from q_lab_toolbox.training_data import mk_training_data_states, measure_rhos
from q_lab_toolbox.readout_operators import _order_n_observables
from q_lab_toolbox.ref.channels import GateBasedChannel

from q_lab_toolbox.ref.unitary_circuits import HardwareEfficientAnsatz

decay_system = DecaySystem(
    ryd_interaction=0.1, omegas=(0.2,), m=1, gammas=(0.3,)
)

H = create_hamiltonian(decay_system)
An = create_jump_operators(decay_system)


rho00 = _rho_rand_haar(m=1, seed=3)
rho10 = _rho_rand_haar(m=1, seed=4)

N = 3
ts = np.linspace(0, 10, N + 1)


rhoss = mk_training_data_states([rho00, rho10], ts, decay_system)


circuit = HardwareEfficientAnsatz(
    m=1, n_qubits=3, depth=2, gate_type="ryd", structure="triangle"
)

Os = _order_n_observables(m=1, n=1)[1:]
print(Os)

training_data = Os, rhoss

training_data2 = (
    Os,
    rhoss[:, 0, :, :],
    np.array([measure_rhos(rhos, Os) for rhos in rhoss]),
)

channel = GateBasedChannel(circuit, -1)

theta_opt, error = channel.run_armijo(training_data2, max_count=2)

print(error)

from q_lab_toolbox.visualize import plot_ess

print(circuit.approximate_evolution(theta_opt, rho00, N))


delta_t = 0.01
taus = np.arange(101)*delta_t
plot_ess(taus, measure_rhos(circuit.approximate_evolution(theta_opt, rho00, 100), Os), ["bla4", "bla3", "bla2","bla1"] )

print(Os)

import matplotlib.pyplot as plt
plt.show()
