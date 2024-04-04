import numpy as np

from q_lab_toolbox.utils.settings import DecaySettings, GateSettings, GeneralSettings
from q_lab_toolbox.target_systems import DecaySettings as DS

target_settings = DS(ryd_interaction=0.2, omegas=(0.5,), m=1, gammas=(0.35,))

ryd_settings = GateSettings(
    cutoff=True,
    depth=10,
    repeats=5,
    n_grad_directions=-1,
    phi=np.pi / 10,
    t_ryd=1.0,
    gammat=0.1,
)

decay_settings = DecaySettings(ryd_interaction=0.2, om0=0.5, om1=0.35)

settings = GeneralSettings(
    seed=6,
    m=1,
    n_training=10,
    t_repeated=5,
    prediction_iterations=20,
    error_type="pauli trace",
    steadystate_weight=0,
    pauli_type="full",
    qubit_structure="triangle d = 0.9",
    lb_type="decay",
    lb_settings=decay_settings,
    circuit_type="ryd",
    circuit_settings=ryd_settings,
    t_lb=1,
    gam0=0.35,
    gam1=0.2,
)
