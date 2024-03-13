from channeler.target_system.settings import (
    DecaySettings,
    RandHaarSettings,
    BasicLinspace,
)

target_settings = DecaySettings(ryd_interaction=0.2, omegas=(0.5,), m=1, gammas=(0.35,))
rho0_settings = RandHaarSettings(m=1, seed=42)

integration_settings = BasicLinspace(t_max=10, n_steps=1000)
