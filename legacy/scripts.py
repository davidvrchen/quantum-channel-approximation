import os, sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import qutip as qt

from legacy.script_utils import time_script

from q_lab_toolbox.initial_states import create_rho0
from q_lab_toolbox.jump_operators import create_jump_operators
from q_lab_toolbox.hamiltonians import create_hamiltonian
from q_lab_toolbox.integration import create_ts

from q_lab_toolbox.readout_operators import (
    create_readout_computational_basis,
    computation_basis_labels,
)

from q_lab_toolbox.training_data import mk_training_data, measure_rhos

from q_lab_toolbox.visualize import plot_ess, compare_ess


@time_script
def solve_lindblad(path):
    # import the settings from the folder
    settingsfile = f"{path}/settings.py"
    sys.path.append(os.path.dirname(os.path.expanduser(settingsfile)))
    import settings

    # read the settings
    target_s = settings.target_system
    rho0_s = settings.rho0
    int_s = settings.integration

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

    target_s = settings.target_system

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
    training_data = settings.training_data

    training_folder = f"{path}/training-data"
    if not os.path.exists(training_folder):
        print(f"folder '{training_folder}' does not exist, making new folder")
        os.makedirs(training_folder)

    ts, Ess, Os = mk_training_data(training_data)

    np.save(f"{path}/training-data/rho0s", training_data.rho0s)
    np.save(f"{path}/training-data/Ess", Ess)
    np.save(f"{path}/training-data/Os", Os)


@time_script
def train_circuit(path):
    # import the settings from the folder
    settingsfile = f"{path}/settings.py"
    sys.path.append(os.path.dirname(os.path.expanduser(settingsfile)))
    import settings

    target_system = settings.target_system
    m = target_system.m
    circuit = settings.circuit
    channel = settings.channel

    training_folder = f"{path}/training-data"
    if not os.path.exists(training_folder):
        print(f"No training data in folder '{training_folder}'")
        return

    rho0s = np.load(f"{path}/training-data/rho0s.npy")
    Ess = np.load(f"{path}/training-data/Ess.npy")
    Os = np.load(f"{path}/training-data/Os.npy")

    Os = [qt.Qobj(O, dims=[[2] * m, [2] * m]) for O in Os]

    training_data = (Os, rho0s, Ess)

    print("starting optimization")
    theta_opt, error = channel.run_armijo(training_data, max_count=50)

    print(theta_opt)

    np.save(f"{path}/flat-theta-opt", circuit.flatten_theta(theta_opt))


@time_script
def plot_approx_channel(path):
    # import the settings from the folder
    settingsfile = f"{path}/settings.py"
    sys.path.append(os.path.dirname(os.path.expanduser(settingsfile)))
    import settings

    rho0 = settings.rho0_test
    target_system = settings.target_system
    circuit = settings.circuit
    training_data = settings.training_data


    path_theta_opt = f"{path}/flat-theta-opt.npy"
    if not os.path.exists(path_theta_opt):
        print(f"No optimized theta found at '{path_theta_opt}'")
        return

    flat_theta = np.load(path_theta_opt)
    theta = circuit.reshape_theta(flat_theta)

    rho0 = create_rho0(rho0)

    N = 100

    rhos = circuit.approximate_evolution(theta, rho0, N)

    Os = create_readout_computational_basis(target_system)
    labels = computation_basis_labels(target_system)
    Ess = measure_rhos(rhos, Os)

    ts = np.arange(N+1) * training_data.delta_t

    plot_ess(ts, Ess, labels)


@time_script
def compare_evolutions(path):
    # import the settings from the folder
    settingsfile = f"{path}/settings.py"
    sys.path.append(os.path.dirname(os.path.expanduser(settingsfile)))
    import settings

    rho0 = settings.rho0
    target_system = settings.target_system
    circuit = settings.circuit
    training_data = settings.training_data

    # get the optimized theta if possible
    path_theta_opt = f"{path}/flat-theta-opt.npy"
    if not os.path.exists(path_theta_opt):
        print(f"No optimized theta found at '{path_theta_opt}'")
        return
    
    flat_theta = np.load(path_theta_opt)
    theta = circuit.reshape_theta(flat_theta)
     
    # get the reference solution
    if not (os.path.exists(f"{path}/ts.npy") and os.path.exists(f"{path}/rhos.npy")):
        print("No solution find, try to run 'solve lindblad' first")
        return

    ts_ref = np.load(f"{path}/ts.npy")
    rhos_ref = np.load(f"{path}/rhos.npy")

    Os = create_readout_computational_basis(target_system)
    labels = computation_basis_labels(target_system)

    Ess_ref = measure_rhos(rhos_ref, Os)

    rho0 = create_rho0(rho0)
    N = 100
    rhos_approx = circuit.approximate_evolution(theta, rho0, N)

    Ess_approx = measure_rhos(rhos_approx, Os)

    ts_approx = np.arange(N+1) * training_data.delta_t

    ref = ts_ref, Ess_ref, "ref"
    approx = ts_approx, Ess_approx, "approx"

    compare_ess(ref, approx, labels)


@time_script
def optimize_theta(path):
    # import the settings from the folder
    settingsfile = f"{path}/settings.py"
    sys.path.append(os.path.dirname(os.path.expanduser(settingsfile)))

    training_folder = f"{path}/training-data"
    if not os.path.exists(training_folder):
        print(f"No training data in folder '{training_folder}'")
        return

    rho0s = np.load(f"{path}/training-data/rho0s.npy")
    Ess = np.load(f"{path}/training-data/Ess.npy")
    Os = np.load(f"{path}/training-data/Os.npy")

    training_data = (Os, rho0s, Ess)

    print(training_data)
