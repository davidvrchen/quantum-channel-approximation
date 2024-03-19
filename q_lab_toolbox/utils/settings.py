"""
Dataclasses used to store necessary settings
to run the algorithm


References:
    original code for decay_examply by @lviss

Notes:
    frozen=True: immutable settings after creation
    slots=True: faster lookup time

Info:
    Created on Thu Mon 26 2024

    @author: davidvrchen
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class GateSettings:
    """
    Dataclass that holds all the necessary settings
    that are related to the gate type of the algorithm

    Args:
    -----

    cutoff (int): cutoff interactions above distance 1 for gate based circuit

    depth (int): depth of simulation circuit (depth-1 entanglement gates)

    repeats (int): number of identical circuits (depth-1), with applying exp(itH)

    n_grad_directions (int): number of parameters to calculate
    the gradient for simultaneous  with stochastic gradient descend.
    -1 for full gradient

    phi (float): initial phi guess (for xy and xy_var)
    t_ryd (float): interaction time for the rydberg entangle gate
    gammat (float): decay rate for decay entangle gate

    """

    cutoff: int = True
    depth: int = 10
    repeats: int = 5
    n_grad_directions: int = -1
    phi: float = np.pi / 10
    t_ryd: float = 1.0
    gammat: float = 0.1


@dataclass(frozen=True, slots=True)
class PulseSettings:
    """
    Dataclass that holds all the necessary settings
    that are related to the pulses of the algorithm

    Args:
    -----

    T_pulse (float): Pulse duration

    driving_H_interaction (str): Hamiltonian of the driving
    interactions (the interactions that are always present ?)
    "basic11";
    "rydberg11"; Rydberg interaction between excited states
    "dipole0110";

    control_H (str): control Hamiltonian
    "rotations";
    "realrotations";

    lambdapar (float): weight on L2 norm of pulse

    Zdt (int): number of segments of the pulse that are optimized separately

    """

    T_pulse: float = 10  # Pulse duration
    driving_H_interaction: str = "rydberg11"
    control_H: str = "rotations+11"
    lambdapar: float = 10 ** (-4)
    Zdt: int = 101


@dataclass(frozen=True, slots=True)
class DecaySettings:
    """
    Dataclass that holds all the necessary settings
    that are related to the pulses of the algorithm

    Args:
    -----

    ryd_interaction (float): Rydberg interaction strength
    between the qubits

    om0 (float): Rabi oscillation frequency qubit 1

    om1 (float): Rabi oscillation frequency qubit 2 (if used)

    om2 (float): Rabi oscillation frequency qubit 3 (if used)


    """

    ryd_interaction: float = 0.2
    om0: float = 0.5
    om1: float = 0.2
    om2: float = 0.2


@dataclass(frozen=True, slots=True)
class TFIMSettings:
    """
    Dataclass that holds all the necessary settings
    that are related to the pulses of the algorithm

    Args:
    -----

    j_en (float): neighbour-neighbour coupling strength

    h_en (float): Transverse magnetic field strength

    """

    j_en: float = 1
    h_en: float = 1


@dataclass(frozen=True, slots=True)
class GeneralSettings:
    """
    Dataclass that holds all the necessary settings
    that are required to run the core algorithm

    Args:
    -----

    seed (int): seed for random number generation with numpy

    m (int): number of qubits

    n_training (int): number of initial rho's to check,
    last one is steady state

    t_repeated (int): number of repeated timesstep per rho

    prediction_iterations (int): number of reaplications
    of the found unitary to check for evolution of errors

    error_type (str): error def'n to use
    "measurement n";
    "pauli trace";
    "bures";
    "trace";
    "wasserstein",
    "trace product";

    steadystate_weight (float): weight given to steady state
    density matrix in calculation of error

    pauli-type (str): Pauli spin matrices to take into account
    "full"; all possible Pauli strings
    "order k"; non-trivial spin matrix for only k qubits
    "random n"; n randomly chosen Pauli strings

    qubit_structure (str): Structure of qubits,
    add d = x to scale the distance between all qubits by x
    "pairs";
    "loose_pairs";
    "triangle";
    "line";

    lb_type (str): type of quantum channel to approx,
    "decay"; Rabi oscillations per qubit and rydberg
    interaction with decay
    "tfim"; transverse field Ising model with decay

    lb_settings (DecaySettings | TFIMSettings): Dataclass
    with parameter settings needed for the ``lb_type``
    quantum channel

    circuit_type (str): entanglement method
    "ryd"; Rydberg gate
    "cnot"; CNOT gate
    "xy";
    "decay";
    "pulse based"; Pulse based

    circuit_settings (GateSettings | PulseSettings): Dataclass
    with parameter settings needed for the ``circuit_type``
    entanglement method

    t_lb (float): Evolution time steps used in calculating
    reference solution

    gam0 (float): decay rate of qubit 1
    gam1 (float): decay rate of qubit 2 (if used)
    gam2 (float): decay rate of qubit 3 (if used)

    max_it_training (int):max number of Armijo steps
    in the gradient descend

    sigmastart (float): starting sigma

    gamma (float): Armijo update criterion

    epsilon (float): finite difference stepsize for gate based gradient

    """

    seed: int

    m: int
    n_training: int
    t_repeated: int
    prediction_iterations: int
    error_type: str
    steadystate_weight: float
    pauli_type: str
    qubit_structure: str

    lb_type: str
    lb_settings: DecaySettings | TFIMSettings
    circuit_type: str
    circuit_settings: GateSettings | PulseSettings

    t_lb: float
    gam0: float = 0.35
    gam1: float = 0.2
    gam2: float = 0.2

    max_it_training: int = 50
    sigmastart: float = 10
    gamma: float = 10 ** (-4)
    epsilon: float = 10 ** (-4)
