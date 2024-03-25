import numpy as np
from Stinespring_unitary_circuits import generate_gate_connections
from q_lab_toolbox.channels import GateBasedChannel
from q_lab_toolbox.error_types import PauliTrace
from q_lab_toolbox.unitary_circuits import HardwareAnsatz, HardwareAnsatzWithH
from decay.settings import settings as s
from decay.settings import target_settings
import q_lab_toolbox.settings as settings
from q_lab_toolbox.hamiltonians import decay_hamiltonian
from q_lab_toolbox.jump_operators import create_jump_operators


def misc():
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

    return train_par, par_dict, entangle_pars


train_par, par_dict, entangle_pars = misc()

pauli_trace = PauliTrace()
hardware_ansatz = HardwareAnsatz(n_qubits=5, depth=10)



channel = GateBasedChannel(
    m=2, circuit=hardware_ansatz, error_type=pauli_trace, par_dict=par_dict
)


target = settings.DecaySettings(
    ryd_interaction=0.2, m=2, omegas=(0.3, 0.5), gammas=(0.2, 0.4)
)

H = decay_hamiltonian(target).full()

An = [operator.full() for operator in create_jump_operators(target)]
channel.set_original_lindblad(H, An, 0.1)


theta = channel.optimize_theta()
