import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import qutip as qt

from channeler.script_utils import time_script

from channeler.target_system.initial_states import create_rho0
from channeler.target_system.jump_operators import create_jump_operators
from channeler.target_system.hamiltonians import create_hamiltonian
from channeler.target_system.qubit_readout_operators import (
    create_readout_individual_qs,
    create_readout_computational_basis,
)
from channeler.target_system.integration import create_ts
from channeler.visualize import (
    plot_evolution_individual_qs,
    plot_evolution_computational_bs,
)

# locate the folder where it all needs to happen
current_dir = os.getcwd()
folder_name = sys.argv[1]
path = os.path.join(current_dir, folder_name)


# import the settings from the folder
settingsfile = f"{path}/settings.py"
sys.path.append(os.path.dirname(os.path.expanduser(settingsfile)))
import settings


def create_observables(s):
    

def plot_evolution(ts, rhoss, pauli_type):
    pass

@time_script
def main() -> None:

    # read the settings
    training_s = settings.training_settings
    target_s = settings.target_settings

    # create initial state, Hamiltonian and jump operators
    rho0 = create_rho0(training_s)
    H = create_hamiltonian(target_s)
    An = create_jump_operators(target_s)

    observables = create_observables(training_s)

    # create ts on which rhos are measured
    ts = create_ts(training_s)

    # numeric integration

    result = qt.mesolve(
        H=H, rho0=rho0, tlist=ts, c_ops=An, e_ops=observables
    )

    # store list of all rhos
    rhoss= result.expect

    # save the result
    np.save(f"{path}/ts", ts)
    np.save(f"{path}/rhoss", rhoss)

    # plot and show the result
    plot_evolution(ts, rhoss, training_s)


if __name__ == "__main__":
    main()
    plt.show()
