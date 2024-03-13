import importlib.util
import os
import sys

import numpy as np

# locate the folder where it all needs to happen
current_dir = os.getcwd()
folder_name = sys.argv[1]
path = os.path.join(current_dir, folder_name)

# import the settings
settingsfile = f"{path}/settings.py"
sys.path.append(os.path.dirname(os.path.expanduser(settingsfile)))

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import settings

s = settings.settings
target_s = settings.target_settings
misc_s = settings.misc_settings

from channeler.stinespring_class import U_circuit, stinespring_unitary_update
from channeler.circuits import generate_gate_connections

from channeler.utils.initial_states import rand_rho_haar

from channeler.target_system.hamiltonians import decay_hamiltonian
from channeler.target_system.jump_operators import jump_operators

from channeler.utils.pauli_matrices import Id, X, Y, Z

# %% Initialization of parameters
save_figs = False  # Save figures as pdf and svg
name = "test run"  # name to prepend to all saved figures


# Seed randomisers
np.random.seed(s.seed)

# Set the initial density matrix
rho0 = rand_rho_haar(s.m)

# Set initial dictionary with arguments
# par_dict = {
#     "qubit_structure": s.qubit_structure,
#     "steadystate_weight": s.steadystate_weight,
# }

# if s.circuit_type == "pulse based":
#     par_dict.update(
#         {
#             "driving_H_interaction": s.circuit_settings.driving_H_interaction,
#             "control_H": s.circuit_settings.control_H,
#             "T_pulse": s.circuit_settings.T_pulse,
#             "lambdapar": s.circuit_settings.lambdapar,
#             "Zdt": s.circuit_settings.Zdt,
#         }
#     )
# else:
#     par_dict.update(
#         {
#             "n_grad_directions": s.circuit_settings.n_grad_directions,
#             "cutoff": s.circuit_settings.cutoff,
#         }
#     )

# # Set entangle gate dictionary
# entangle_pars = {
#     "t_ryd": s.circuit_settings.t_ryd,
#     "phi": s.circuit_settings.phi,
#     "gammat": s.circuit_settings.gammat,
# }

# Create correct parameter array that will be optimized
theta0 = (
    np.ones([s.circuit_settings.depth, 2 * s.m + 1, 3]) * np.pi / 2
)  # Initial theta guess
pars_per_layer = len(
    generate_gate_connections(2 * s.m + 1, structure=s.qubit_structure, cutoff=True)
)
gate_par = 0
if s.circuit_type == "xy":
    phi0 = np.ones([s.circuit_settings.depth, pars_per_layer]) * s.circuit_settings.phi
    gate_par = phi0
    theta0 = np.concatenate((np.ravel(theta0), np.ravel(phi0)))
elif s.circuit_type == "ryd":
    t_ryd0 = (
        np.ones(
            [
                s.circuit_settings.depth,
            ]
        )
        * s.circuit_settings.t_ryd
    )
    gate_par = t_ryd0
    theta0 = np.concatenate((np.ravel(theta0), np.ravel(t_ryd0)))
elif s.circuit_type == "decay":
    gammat0 = (
        np.ones([s.circuit_settings.depth, pars_per_layer]) * s.circuit_settings.gammat
    )
    gate_par = s.circuit_settings.gammat0
    theta0 = np.concatenate((np.ravel(theta0), np.ravel(gammat0)))
elif s.circuit_type == "pulse based":
    if "+11" in s.circuit_settings.control_H:
        n_controls_H = 2 * (2 * s.m + 1)
    else:
        n_controls_H = 2 * s.m + 1
    theta0 = np.zeros((n_controls_H, s.circuit_settings.Zdt, 2)) + 0.02
    # =============================================================================
    #     theta0[0,:,0] = np.sin(np.linspace(0,T_pulse, Zdt) + np.random.rand()*2*np.pi) *0.1*np.random.rand()
    #     theta0[0,:,1] = np.sin(np.linspace(0,T_pulse, Zdt) + np.random.rand()*2*np.pi) *0.1*np.random.rand()
    #     theta0[1,:,0] = np.sin(np.linspace(0,T_pulse, Zdt) + np.random.rand()*2*np.pi) *0.1*np.random.rand()
    #     theta0[1,:,1] = np.sin(np.linspace(0,T_pulse, Zdt) + np.random.rand()*2*np.pi) *0.1*np.random.rand()
    #     theta0[3,:,0] = np.sin(np.linspace(0,T_pulse, Zdt) + np.random.rand()*2*np.pi) *0.1*np.random.rand()
    #     theta0[3,:,1] = np.sin(np.linspace(0,T_pulse, Zdt) + np.random.rand()*2*np.pi) *0.1*np.random.rand()
    # =============================================================================
    theta0 = np.ravel(theta0)
else:
    theta0 = np.ravel(theta0)

# Set parameter dictionaries
# train_par = {
#     "n_training": s.n_training,
#     "seed": s.seed,
#     "depth": s.circuit_settings.depth,
#     "theta0": theta0,
#     "max_it_training": s.max_it_training,
#     "epsilon": s.epsilon,
#     "gamma": s.gamma,
#     "sigmastart": s.sigmastart,
#     "circuit_type": s.circuit_type,
#     "pauli_type": s.pauli_type,
#     "t_repeated": s.t_repeated,
# }


# # %% Initialize the class
# print(train_par)
# print(par_dict)
# print(
#     s.lb_type,
#     s.lb_settings.om0,
#     s.lb_settings.om1,
#     s.gam0,
#     s.gam1,
# )


stinespring_class = stinespring_unitary_update(s, misc_s=misc_s, split_H=True)

H = decay_hamiltonian(target_s).full()
An = [operator.full() for operator in jump_operators(target_s)]
stinespring_class.set_original_lindblad(H, An, s.t_lb)


# %% Set random initial rho and their evolutions under U or Lindblad. Eg, make trainingdata

stinespring_class.set_training_data(
    s.n_training, s.seed, paulis=s.pauli_type, t_repeated=s.t_repeated
)


t0 = 0
t1 = s.t_lb
steps = 100
t_range = np.linspace(t0, t1, steps)
sol = np.zeros([steps, 2**s.m, 2**s.m])
for i, ti in enumerate(t_range):
    stinespring_class.set_original_lindblad(H, An, ti)
    sol[i] = np.real(stinespring_class.evolution(rho0))

qubit_strings = np.array(["" for _ in range(2**s.m)], dtype=f"<U{s.m}")
for i in range(2**s.m):
    qubit_strings[i] = format(i, f"0{s.m}b")

plt.figure()
for i in range(2**s.m):
    for j in range(2**s.m):
        plt.plot(
            t_range,
            sol[:, i, j],
            label=rf"$|{qubit_strings[i]}\rangle \langle{qubit_strings[j]}|$",
        )
plt.legend(loc=0, bbox_to_anchor=(1.0, 1.0))
plt.xlim([0, s.t_lb])

plt.figure()
for i in range(2**s.m):
    plt.plot(
        t_range,
        sol[:, i, i],
        label=rf"$|{qubit_strings[i]}\rangle \langle{qubit_strings[i]}|$",
    )
plt.legend()
plt.xlim([0, s.t_lb])

print("Evolution tested")


# %% Training of unitary circuit that approximates the exact evolution
# %% Set random initial rho and their evolutions under U or Lindblad. Eg, make trainingdata


def optimzed_circuit():
    """Check whether there exists an optimzed circuit."""
    return os.path.exists(f"{path}/gate_par.npy") and os.path.exists(
        f"{path}/theta_par.npy"
    )


if not optimzed_circuit():
    stinespring_class.set_unitary_circuit(
        circuit_type=s.circuit_type, depth=s.circuit_settings.depth, gate_par=gate_par
    )
    print("Initial error: ", stinespring_class.training_error(theta0))

    theta1, error1 = stinespring_class.run_all_lindblad(H, An, s)

    theta_opt, gate_par_opt = stinespring_class.reshape_theta_phi(
        stinespring_class.theta_opt
    )

    np.save(f"{path}/theta", theta_opt)
    np.save(f"{path}/gate_par", gate_par_opt)

    print("Unitary trained")


theta_opt = np.load(f"{path}/theta.npy")
gate_par_opt = np.load(f"{path}/gate_par.npy")
