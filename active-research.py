import numpy as np
from q_lab_toolbox.channels import GateBasedChannel
from q_lab_toolbox.unitary_circuits import HardwareAnsatz
from q_lab_toolbox.hamiltonians import decay_hamiltonian
from q_lab_toolbox.jump_operators import create_jump_operators
from training_data import mk_training_data2
from q_lab_toolbox.target_systems import DecaySystem
from q_lab_toolbox.initial_states import rho_rand_haar, RhoRandHaar
from q_lab_toolbox.readout_operators import order_n_observables, OrdernObservables, create_observables


hardware_ansatz = HardwareAnsatz(
    m=2,
    n_qubits=5,
    depth=10,
    gate_type="ryd",
    structure="triangle",
    n_repeats=5
)


channel = GateBasedChannel(circuit=hardware_ansatz, n_grad_direction=-1)


target = DecaySystem(
    ryd_interaction=0.2, m=2, omegas=(0.3, 0.5), gammas=(0.2, 0.4)
)

# lindblad
H = decay_hamiltonian(target)
An = create_jump_operators(target)

# initial state
rho0_s = RhoRandHaar(m=2, seed=5)
rho0 = rho_rand_haar(rho0_s)

observables = OrdernObservables(m=2, n=1)
Os = create_observables(observables)

data = mk_training_data2(rho0=rho0, delta_t=0.1, n_training=3, Os=Os, s=target)


theta = channel.optimize_theta(training_data=data)
print(theta)
