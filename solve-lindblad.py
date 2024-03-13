"""
HOW THE SCRIPT WORKS
To run the script use:

>>> python solve-lindblad.py "folder-name"

the folder should contain a file called "settings.py"
where all the necessary settings are def'd.

To solve the Lindbladian three settings are required,
they respectively define the:
    1) target system
    2) initial state rho0
    3) choices relating to numeric
    integration of Lindbladian

Based on these settings the Lindbladian is integrated,
the tuple (ts, rhos) is the calculated reference solution.
This result is stored in two .npy files,
"ts.npy" and "rhos.npy" respectively.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import qutip as qt

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

# information message for the user
print(
    f"""\n
Starting script "solve-lindblad.py" with settings def'd in\r
    {sys.argv[1]}/settings.py\r
Expecting three settings objects:\r
    target_settings,\r
    rho0_settings,\r
    integration_settings\n
the result from numeric integration will be stored in:\r
    "{sys.argv[1]}/ts.npy", "{sys.argv[1]}/rhoss_indiv.npy" and "{sys.argv[1]}/rhoss_comp.npy"\r
which store the ts, rhos measured in individual qubit basis and \r
rhos measured in computational basis respectively.
\n"""
)


# read the settings
target_s = settings.target_settings
rho0_s = settings.rho0_settings
int_s = settings.integration_settings


# create initial state, Hamiltonian and jump operators
rho0 = create_rho0(rho0_s)
H = create_hamiltonian(target_s)
An = create_jump_operators(target_s)


readout_indiv_qs = create_readout_individual_qs(target_s)
readout_comp_bs = create_readout_computational_basis(target_s)


# create ts on which rhos are measured
ts = create_ts(int_s)

# numeric integration

result_indiv_qs = qt.mesolve(H=H, rho0=rho0, tlist=ts, c_ops=An, e_ops=readout_indiv_qs)
result_comp_bs = qt.mesolve(H=H, rho0=rho0, tlist=ts, c_ops=An, e_ops=readout_comp_bs)

# store list of all rhos
rhoss_indiv = result_indiv_qs.expect
rhoss_comp = result_comp_bs.expect

# save the result
np.save(f"{path}/ts", ts)
np.save(f"{path}/rhoss_indiv", rhoss_indiv)
np.save(f"{path}/rhoss_comp", rhoss_comp)


# plot and show the result
plot_evolution_individual_qs(ts, rhoss_indiv)
plot_evolution_computational_bs(ts, rhoss_comp)
plt.show()
