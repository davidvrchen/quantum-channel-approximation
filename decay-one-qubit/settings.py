from q_lab_toolbox.settings import (
    DecaySettings,
    RandHaarSettings,
    BasicLinspace,
)

target_settings = DecaySettings(ryd_interaction=0.2, omegas=(0.5,), m=1, gammas=(0.35,))
rho0_settings = RandHaarSettings(m=1, seed=42)

integration_settings = BasicLinspace(t_max=10, n_steps=1000)


from dataclasses import dataclass


@dataclass
class TrainingSettings:
    pauli_type: str
    order: int


training_settings = TrainingSettings(pauli_type="full", order=1)
