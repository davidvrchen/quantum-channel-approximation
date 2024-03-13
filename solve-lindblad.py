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
from channeler.target_system.qubit_readout_operators import create_readout_ops


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
the result from numeric integration will be stored in:\r
    "{sys.argv[1]}/ts.npy" and "{sys.argv[1]}/rhoss.npy"\r
respectively
\n"""
)


# read the settings
target_s = settings.target_settings
rho0_s = settings.rho0_settings
# integration_s = settings.integration_settings


# create initial state, Hamiltonian and jump operators
rho0 = create_rho0(rho0_s)
H = create_hamiltonian(target_s)
An = create_jump_operators(target_s)
readout_ops = create_readout_ops(target_s)


# create ts on which rhos are measured
ts = np.linspace(0, 20, 1000)

# numeric integration
result = qt.mesolve(H=H, rho0=rho0, tlist=ts, c_ops=An, e_ops=readout_ops)

# store list of all rhos
rhoss = result.expect


# save the result
np.save(f"{path}/ts", ts)
np.save(f"{path}/rhoss", rhoss)


def plot_evolution(ts: np.ndarray, rhoss: [np.ndarray]) -> plt.axes:
    """Plots the evolution of all rhos as a function of ts
    with some basic formatting.

    Args:
        ts (np.ndarray): times t_i
        rhoss (list[np.ndarray]): rho_i at time t_i
    """

    fig, ax = plt.subplots()

    for i, rhos in enumerate(rhoss):
        state = i % 2
        ax.plot(ts, rhos, label=rf"$q_{i//2} : |{state}\rangle \langle{state}|$")

    # some formatting to make plot look nice
    plt.ylabel("population")
    plt.xlabel("time")
    plt.ylim(0, 1)
    plt.legend()
    
    return ax


# show the result
plot_evolution(ts, rhoss)
plt.show()
