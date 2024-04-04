from q_lab_toolbox.target_systems import (
    DecaySystem,
    RhoRandHaar,
    BasicLinspace,
)

target_settings = DecaySystem(
    ryd_interaction=0.1, omegas=(0.5, 0.3), m=2, gammas=(0.3, 0.2)
)
rho0_settings = RhoRandHaar(m=2, seed=42)

integration_settings = BasicLinspace(t_max=10, n_steps=1000)
