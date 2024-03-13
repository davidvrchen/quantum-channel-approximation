import importlib.util
import os
import sys

import numpy as np

# locate the folder where it all needs to happen
current_dir = os.getcwd()
folder_name = sys.argv[1]
path = os.path.join(current_dir, folder_name)

# import the settings
settingsfile = f'{path}/settings.py'
sys.path.append(os.path.dirname(os.path.expanduser(settingsfile)))

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import settings

s = settings.settings 
target_s = settings.target_settings

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
par_dict = {
    "qubit_structure": s.qubit_structure,
    "steadystate_weight": s.steadystate_weight,
}

if s.circuit_type == "pulse based":
    par_dict.update(
        {
            "driving_H_interaction": s.circuit_settings.driving_H_interaction,
            "control_H": s.circuit_settings.control_H,
            "T_pulse": s.circuit_settings.T_pulse,
            "lambdapar": s.circuit_settings.lambdapar,
            "Zdt": s.circuit_settings.Zdt,
        }
    )
else:
    par_dict.update(
        {
            "n_grad_directions": s.circuit_settings.n_grad_directions,
            "cutoff": s.circuit_settings.cutoff,
        }
    )

# Set entangle gate dictionary
entangle_pars = {
    "t_ryd": s.circuit_settings.t_ryd,
    "phi": s.circuit_settings.phi,
    "gammat": s.circuit_settings.gammat,
}

# Create correct parameter array that will be optimized
theta0 = (
    np.ones([s.circuit_settings.depth, 2 * s.m + 1, 3]) * np.pi / 2
)  # Initial theta guess
pars_per_layer = len(
    generate_gate_connections(2 * s.m + 1, structure=s.qubit_structure, cutoff=True)
)
gate_par = 0
if s.circuit_type == "xy":
    phi0 = (
        np.ones([s.circuit_settings.depth, pars_per_layer]) * s.circuit_settings.phi
    )
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
        np.ones([s.circuit_settings.depth, pars_per_layer])
        * s.circuit_settings.gammat
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
train_par = {
    "n_training": s.n_training,
    "seed": s.seed,
    "depth": s.circuit_settings.depth,
    "theta0": theta0,
    "max_it_training": s.max_it_training,
    "epsilon": s.epsilon,
    "gamma": s.gamma,
    "sigmastart": s.sigmastart,
    "circuit_type": s.circuit_type,
    "pauli_type": s.pauli_type,
    "t_repeated": s.t_repeated,
}


# %% Initialize the class
print(train_par)
print(par_dict)
print(
    s.lb_type,
    s.lb_settings.om0,
    s.lb_settings.om1,
    s.gam0,
    s.gam1,
)



stinespring_class = stinespring_unitary_update(
    s.m, error_type=s.error_type, circuit_type=s.circuit_type, split_H=True, par_dict=par_dict
)

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
    return os.path.exists(f"{path}/gate_par.npy") and os.path.exists(f"{path}/theta_par.npy")

if not optimzed_circuit():
    stinespring_class.set_unitary_circuit(
        circuit_type=s.circuit_type, depth=s.circuit_settings.depth, gate_par=gate_par
    )
    print("Initial error: ", stinespring_class.training_error(theta0))

    theta1, error1 = stinespring_class.run_all_lindblad(
        H, An, s.t_lb, **train_par, **entangle_pars
    )

    theta_opt, gate_par_opt = stinespring_class.reshape_theta_phi(
        stinespring_class.theta_opt
    )

    np.save(f"{path}/theta", theta_opt)
    np.save(f"{path}/gate_par", gate_par_opt)

    print("Unitary trained")


theta_opt = np.load(f"{path}/theta.npy")
gate_par_opt = np.load(f"{path}/gate_par.npy")


plt.figure()
plt.plot(error1[1:])
plt.yscale("log")
plt.ylabel(f"Error - {s.error_type}")
plt.xlabel("Iteration")
if save_figs:
    plt.savefig(f"Figures/{name}.svg", bbox_inches="tight")
    plt.savefig(f"Figures/{name}.pdf", bbox_inches="tight")

if s.circuit_type == "pulse based":
    theta1 = stinespring_class.reshape_theta_phi(theta1)[0]
    plt.figure()
    for k in range(theta1.shape[0]):
        colours = ["b", "r", "g", "m", "y", "k"]
        plt.plot(
            np.linspace(0, s.circuit_settings.T_pulse, s.circuit_settings.Zdt),
            theta1[k, :, 0],
            f"{colours[k%6]}-",
            label=f"qubit {k}",
        )
        plt.plot(
            np.linspace(0, s.circuit_settings.T_pulse, s.circuit_settings.Zdt),
            theta1[k, :, 1],
            f"{colours[k%6]}--",
        )
        plt.legend()
    if save_figs:
        plt.savefig(f"Figures/{name} Pulse.svg", bbox_inches="tight")


# %% Reapplying unitary to new data
# prediction_iterations = 50

# Set new rho0
stinespring_class.set_training_data(
    s.n_training, s.seed + 1, paulis=s.pauli_type, t_repeated=s.t_repeated
)

# Pick one rho by index for plotting
rho_i = 0

# Initialize empty arrays
error = np.zeros(s.prediction_iterations)
ev_exact = np.zeros(
    [s.n_training, s.prediction_iterations, 2**s.m, 2**s.m], dtype=np.complex128
)
ev_exact_full = np.zeros([100, 2**s.m, 2**s.m], dtype=np.complex128)
ev_circuit = np.zeros(
    [s.n_training, s.prediction_iterations, 2**s.m, 2**s.m], dtype=np.complex128
)
trace_dist = np.zeros([s.n_training, s.prediction_iterations])

for i, rho in enumerate(stinespring_class.training_data[0]):
    ev_exact[i] = stinespring_class.evolution_n(s.prediction_iterations, rho)[1:]
    ev_circuit[i] = stinespring_class.unitary_approx_n(s.prediction_iterations, rho)[1:]

    for n in range(s.prediction_iterations):
        ev_exact_root = sc.linalg.sqrtm(ev_exact[i, n])

        trace_dist[i, n] = max(
            0,
            1
            - np.abs(
                np.trace(
                    sc.linalg.sqrtm(ev_exact_root @ ev_circuit[i, n] @ ev_exact_root)
                )
            ),
        )

error = np.einsum("ij->j", trace_dist) / s.n_training

ev_exact_full = np.real(
    stinespring_class.evolution_t(
        np.linspace(0, s.prediction_iterations * s.t_lb, 200),
        stinespring_class.training_data[0, rho_i],
    )
)

colours = ["b", "r", "g", "m", "y", "k"]
plt.figure()
x_exact = np.linspace(0, s.prediction_iterations * s.t_lb, 200)
x_approx = np.array(range(1, (s.prediction_iterations + 1))) * s.t_lb
for i in range(2**s.m):
    plt.plot(
        x_exact,
        ev_exact_full[:, i, i],
        f"{colours[i%6]}-",
        label=rf"$|{qubit_strings[i]}\rangle \langle{qubit_strings[i]}|$",
    )
    plt.plot(x_approx, np.real(ev_circuit[rho_i, :, i, i]), f"{colours[i%6]}x")
    plt.plot(
        np.linspace(0, s.prediction_iterations * s.t_lb, 3),
        np.zeros(3) + np.real(stinespring_class.steady_state[i, i]),
        f"{colours[i%6]}--",
    )
plt.legend(loc="upper right")
plt.xlabel("System evolution time")
plt.ylabel("Population")
plt.xlim([0, s.prediction_iterations * s.t_lb])
# plt.ylim([-0.2, 1.2])
if save_figs:
    plt.savefig(f"Figures/{name} prediction single rho.pdf", bbox_inches="tight")


plt.figure()
for i in range(2**s.m):
    plt.plot(
        x_approx,
        np.real(ev_exact[rho_i, :, i, i] - ev_circuit[rho_i, :, i, i]),
        f"{colours[i%6]}o",
        label=rf"$|{qubit_strings[i]}\rangle \langle{qubit_strings[i]}|$",
    )

plt.plot(x_exact, np.zeros(200), "k--")
plt.legend()
plt.xlabel("System evolution time")
plt.ylabel("Population error")
plt.xlim([0, s.prediction_iterations * s.t_lb])
if save_figs:
    plt.savefig(f"Figures/{name} prediction single rho error.pdf", bbox_inches="tight")

plt.figure()
plt.plot(range(1, s.prediction_iterations + 1), error)
plt.xlabel("Repetitions of U")
plt.ylabel("Error - Bures")
plt.xlim([1, s.prediction_iterations])
if save_figs:
    plt.savefig(f"Figures/{name} predictions.svg", bbox_inches="tight")
    plt.savefig(f"Figures/{name} predictions.pdf", bbox_inches="tight")


print(stinespring_class.evolution(np.identity(2**s.m)))

plt.show()
