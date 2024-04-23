from q_lab_toolbox.target_systems import DecaySystem
from q_lab_toolbox.initial_states import RhoRandHaar
from q_lab_toolbox.integration import BasicLinspace

target_system = DecaySystem(
    ryd_interaction=0.1, omegas=(0.5, 0.3), m=2, gammas=(0.3, 0.2)
)

rho0 = RhoRandHaar(m=2, seed=42)

integration = BasicLinspace(t_max=10, n_steps=1000)
