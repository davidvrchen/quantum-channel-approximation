from q_lab_toolbox.target_systems import DecaySystem
from q_lab_toolbox.initial_states import RhoRandHaar
from q_lab_toolbox.integration import BasicLinspace
from q_lab_toolbox.training_data import RandomTrainingData
from q_lab_toolbox.readout_operators import OrdernObservables
from q_lab_toolbox.ref.unitary_circuits import HardwareEfficientAnsatz
from q_lab_toolbox.ref.channels import GateBasedChannel


target_system = DecaySystem(ryd_interaction=0.2, omegas=(0.5, 1.2), m=2, gammas=(0.35, 1.2))

rho0 = RhoRandHaar(m=2, seed=42)

integration = BasicLinspace(t_max=10, n_steps=1000)

Os = OrdernObservables(2, 2)

training_data = RandomTrainingData(
    target_system=target_system, N=2, delta_t=0.1, seed=42, L=10, Os=Os
)

circuit = HardwareEfficientAnsatz(
    m=2, n_qubits=5, depth=2, gate_type="ryd", structure="triangle"
)

channel = GateBasedChannel(circuit, -1)