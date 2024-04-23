import os, sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import qutip as qt

from q_lab.script_utils import time_script

from q_lab_toolbox.initial_states import create_rho0
from q_lab_toolbox.jump_operators import create_jump_operators
from q_lab_toolbox.hamiltonians import create_hamiltonian
from q_lab_toolbox.integration import create_ts

from q_lab_toolbox.readout_operators import (
    order_n_pauli_strs,
    _order_n_observables,
    order_n_pauli_strs,
    create_readout_computational_basis,
    computation_basis_labels,
)

from q_lab_toolbox.training_data import mk_training_data_states, measure_rhos

from q_lab_toolbox.visualize import plot_ess


@time_script
def solve_lindblad(path):
    # import the settings from the folder
    settingsfile = f"{path}/settings.py"
    sys.path.append(os.path.dirname(os.path.expanduser(settingsfile)))
    import settings

    # read the settings
    target_s = settings.target_settings
    rho0_s = settings.rho0_settings
    int_s = settings.integration_settings

    # create initial state, Hamiltonian and jump operators
    rho0 = create_rho0(rho0_s)
    H = create_hamiltonian(target_s)
    An = create_jump_operators(target_s)

    # create ts on which rhos are measured
    ts = create_ts(int_s)

    # numeric integration
    result = qt.mesolve(H=H, rho0=rho0, tlist=ts, c_ops=An)

    # store list of all rhos
    rhos = result.states

    # save the result
    np.save(f"{path}/ts", ts)
    np.save(f"{path}/rhos", rhos)

    # plot the result
    labels = computation_basis_labels(target_s)
    Os = create_readout_computational_basis(target_s)
    Ess = measure_rhos(rhos, Os)
    plot_ess(ts, Ess, labels)


@time_script
def plot_reference_solution(path):
    settingsfile = f"{path}/settings.py"
    sys.path.append(os.path.dirname(os.path.expanduser(settingsfile)))
    import settings

    target_s = settings.target_settings

    # load the result
    if not (os.path.exists(f"{path}/ts.npy") and os.path.exists(f"{path}/rhos.npy")):
        print("No solution find, try to run 'solve lindblad' first")
        return

    ts = np.load(f"{path}/ts.npy")
    rhos = np.load(f"{path}/rhos.npy")

    # plot the result
    labels = computation_basis_labels(target_s)
    Os = create_readout_computational_basis(target_s)
    Ess = measure_rhos(rhos, Os)
    plot_ess(ts, Ess, labels)


@time_script
def make_training_data(path):
    # import the settings from the folder
    settingsfile = f"{path}/settings.py"
    sys.path.append(os.path.dirname(os.path.expanduser(settingsfile)))
    import settings

    # read the settings
    target_s = settings.target_settings
    training_s =settings.training_settings

    # create Hamiltonian and jump operators
    H = create_hamiltonian(target_s)
    An = create_jump_operators(target_s)
