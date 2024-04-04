from argparse import ArgumentParser
import sys, os

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import qutip as qt

from q_lab_utils.script_utils import time_script

from q_lab_toolbox.initial_states import create_rho0
from q_lab_toolbox.jump_operators import create_jump_operators
from q_lab_toolbox.hamiltonians import create_hamiltonian
from q_lab_toolbox.qubit_readout_operators import (
    create_readout_individual_qs,
    create_readout_computational_basis,
)
from q_lab_toolbox.integration import create_ts
from q_lab_toolbox.visualize import (
    plot_evolution_individual_qs,
    plot_evolution_computational_bs,
)

from q_lab_utils.script_utils import time_script

# setup the argument parsing for CLI
parser = ArgumentParser(prog="q-lab")

parser.add_argument(
    "-f", "--folder", type=str, help="enter name of folder to operate on", required=True
)
parser.add_argument(
    "-a",
    "--action",
    metavar="OPTION",
    type=str,
    help="action to perform in the specified folder",
    choices=["solve lindblad", "mk settings", "mk training data"],
    required=True,
)

args = parser.parse_args()


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

    readout_indiv_qs = create_readout_individual_qs(target_s)
    readout_comp_bs = create_readout_computational_basis(target_s)

    # create ts on which rhos are measured
    ts = create_ts(int_s)

    # numeric integration

    result_indiv_qs = qt.mesolve(
        H=H, rho0=rho0, tlist=ts, c_ops=An, e_ops=readout_indiv_qs
    )
    result_comp_bs = qt.mesolve(
        H=H, rho0=rho0, tlist=ts, c_ops=An, e_ops=readout_comp_bs
    )

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


@time_script
def make_settings():
    pass


@time_script
def make_training_data():
    pass




def main() -> None:

    # locate the folder where it all needs to happen
    current_dir = os.getcwd()
    folder_name = args.folder
    path = os.path.join(current_dir, folder_name)

    if not os.path.exists(path):
        print(f"folder {folder_name} does not exist, making new folder")
        os.makedirs(path)

    # make settings file
    match args.action:
        case "solve lindblad":
            solve_lindblad(path=path)
            plt.show()
        case "mk settings":
            make_settings()

    print("hello world")


if __name__ == "__main__":

    main()
    plt.show()
