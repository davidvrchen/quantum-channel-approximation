from q_lab_toolbox.target_systems import DecaySystem
from q_lab_toolbox.initial_states import RhoRandHaar
from q_lab_toolbox.integration import BasicLinspace


target_settings = DecaySystem(ryd_interaction=0.2, omegas=(0.5,), m=1, gammas=(0.35,))
rho0_settings = RhoRandHaar(m=1, seed=42)

integration_settings = BasicLinspace(t_max=10, n_steps=1000)


from dataclasses import dataclass


@dataclass
class TrainingSettings:
    pauli_type: str
    order: int


training_settings = TrainingSettings(pauli_type="full", order=1)
