# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:35:01 2023

@author: lviss
"""

import numpy as np
import torch as to
import qutip as qt
import scipy as sc
import matplotlib.pyplot as plt

from my_functions import generate_hamiltonian, get_paulis, ham_expand_pauli, \
    exp_approx, unitary, unitary_trot
    
from stinespring_t_update_classes import stinespring_unitary_update, U_circuit

from Stinespring_unitary_circuits import generate_gate_connections

#%% Initialization of parameters
save_figs = False                   # Save figures as pdf and svg
name = 'test run'                   # name to prepend to all saved figures

# General parameters
m = 2
n_training = 10                     # Number of initial rho's to check 
nt_training = 4                     # Number of repeated timesteps per rho
prediction_iterations = 20          # Number of reaplications of the found unitary to check for evolution of errors
seed = 3                            # Seed for random initial rho's
error_type = 'pauli trace'          # Basis for error: "measurement n", "pauli trace", "bures", "trace", 'wasserstein', 'trace product' 
steadystate_weight = 10              # Weight given to steady state density matrix in calculation of error
pauli_type = 'full'              # Pauli spin matrices to take into account. 
                                    # Options: 'full', 'order k' for k-local, 'random n'
                                    
circuit_type = 'pulse based'            # Gate type used to entangle, 
                                    #   choose: cnot, ryd, xy, decay, with varied parameters
                                    # choose: 'pulse based'
qubit_structure = 'triangle d = 0.9'        # structure of qubits: pairs, loose_pairs, triangle, line
                                    # add d = some number to scale the distance between all qubits

# Gate based circuit parameters
cutoff = True                       # Cutoff interactions above distance 1 for gate based circuit
depth = 3                           # Depth of simulation circuit (depth-1 entanglement gates)
repeats = 5                         # Number of identical circuits (depth-1), with applying exp(itH)
n_grad_directions = 10              # Number of parameters to calculate the gradient for simultaneous 
                                    # (for stochastic gradient descend), set to -1 for full gradient

phi = np.pi/10                      # Initial phi guess (for xy and xy_var)
t_ryd = 1.0                           # Interaction time for the rydberg entangle gate
gammat = 0.1                        # Decay rate for decay entangle gate


# Pulse based parameters
T_pulse = 5                         # Pulse duration 
driving_H_interaction = 'rydberg11'   # basic11, rydberg11, dipole0110
control_H = 'rotations+11'             # Control Hamiltonian ('rotations' or 'realrotations')
lambdapar = 0 # 10**(-4)
Zdt = 101


# Armijo gradient descend parameters
max_it_training = 100   # Max number of Armijo steps in the gradient descend
sigmastart = 1          # Starting sigma
gamma = 10**(-4)        # Armijo update criterion
epsilon = 10**(-4)      # Gradient stepsize

# Quantum channel defined by a Lindbladian or by another unitary circuit
from_lindblad = True

# Lindblad equation parameters
lb_type = 'decay' # Type of quantum channel to approx, 
                    # 'decay' is decay + H of sigma_x per qubit and rydberg interaction
                    # 'tfim' is transverse field ising model with decay
t_lb = 0.5       # Evolution time
gam0 = 0.2      # Decay rate qubit 1
gam1 = 0.2      # Decay rate qubit 2
gam2 = 0.2      # Decay rate qubit 3
om0 = 0.5         # Hamiltonian forcing strength qubit 1
om1 = 0.5        # Hamiltonian forcing strength qubit 2
om2 = 0.35      # Hamiltonian forcing strength qubit 3
ryd_interaction = 0 # 0.2 #Rydberg interaction strength between the qubits

j_en = 1    # neighbour-neighbour coupling strength for transverse field ising model
h_en = 1    # Transverse magnetic field strength

#### Set parameter dependent things ####

# Seed randomisers
np.random.seed(seed)

# rho0, used for plotting the evolution of the Lindblad equation (if used)
# |11><11| state
rho0 = np.zeros([2**m,2**m])
rho0[3,3] = 1

# Fully mixed
rho0 = np.eye(2**m)/2**m

# Random start
random_ket = qt.rand_ket_haar(dims = [[2**m], [1]])
random_ket.dims = [[2]*m,[2]*m]
random_bra = random_ket.dag()
rho0 = (random_ket * random_bra).full()

# Set initial dictionary with arguments
par_dict = {'qubit_structure': qubit_structure, 'steadystate_weight': steadystate_weight}
if circuit_type == 'pulse based':
    par_dict.update({'driving_H_interaction': driving_H_interaction,
            'control_H': control_H, 'T_pulse': T_pulse,
            'lambdapar': lambdapar, 'Zdt': Zdt})
else:
    par_dict.update({'n_grad_directions': n_grad_directions, 'cutoff': cutoff})

# Set entangle gate dictionary
entangle_pars = {'t_ryd': t_ryd, 'phi': phi, 'gammat': gammat}

# Modify theta0 to also include phi0
theta0 = np.ones([depth, 2*m+1, 3])*np.pi/2    # Initial theta guess
pars_per_layer = len(generate_gate_connections(2*m+1, structure = qubit_structure, cutoff = True))
gate_par = 0
if circuit_type == 'xy':
    phi0 = np.ones([depth-1, pars_per_layer])*phi
    gate_par = phi0
    theta0 = np.concatenate((np.ravel(theta0),np.ravel(phi0)))
elif circuit_type == 'ryd':
    t_ryd0 = np.ones([depth-1,])*t_ryd
    gate_par = t_ryd0
    theta0 = np.concatenate((np.ravel(theta0),np.ravel(t_ryd0)))
elif circuit_type == 'decay':
    gammat0 = np.ones([depth-1, pars_per_layer])*gammat
    gate_par = gammat0
    theta0 = np.concatenate((np.ravel(theta0),np.ravel(gammat0)))
elif circuit_type == 'pulse based':
    if '+11' in control_H:
        n_controls_H = 2*(2*m+1)
    else:
        n_controls_H = 2*m+1
    theta0 = np.zeros((n_controls_H, Zdt, 2))+0.02
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
train_par = {'n_training':n_training, 'seed': seed, 'depth':depth, 'theta0':theta0, 
             'max_it_training':max_it_training, 'epsilon':epsilon, 'gamma':gamma, 
             'sigmastart':sigmastart,'circuit_type':circuit_type, 'pauli_type':pauli_type,
             't_repeated': nt_training}


#%% Initialize the class
stinespring_class = stinespring_unitary_update(m, error_type = error_type, circuit_type = circuit_type, par_dict = par_dict)

#%% Define evolution operator
# Pauli spin matrices
Id = np.array([[1,0],[0,1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,1j],[-1j,0]])
Z = np.array([[1,0],[0,-1]])

if from_lindblad:
    # Lindbladian
    if m==1:
        An = np.array([[[0,gam0**(1/2)],[0,0]]])
        H = om0*np.kron(X,Id)
    elif m==2:
        An = np.array([[[0,gam0**(1/2),0,0],[0,0,0,0],[0,0,0,gam0**(1/2)],[0,0,0,0]], #|. 1> to |. 0>
                      [[0,0,gam1**(1/2),0],[0,0,0,gam1**(1/2)],[0,0,0,0],[0,0,0,0]], #|1 .> to |0 .>
                      ])
        
        if lb_type =='decay':
            # Rabi osscilation Hamiltonian + rydberg interaction
            H = (om0*np.kron(X,Id) + om1*np.kron(Id,X) + ryd_interaction*np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]))
        
        elif lb_type == 'tfim':
            # Transverse field Ising model Hamiltonian
            H = j_en *(np.kron(Z, Id) @np.kron(Id, Z)) \
                - h_en *(np.kron(X,Id)+np.kron(Id,X))
                
        else:
            print("Lindblad type {} invalid, quantum channel contains only simple decay".format(lb_type))
            
    elif m==3:
        gam0 = gam0**(1/2)
        gam1 = gam1**(1/2)
        gam2 = gam2**(1/2)
        An = np.array([
            [[0, gam0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,gam0,0,0,0,0], [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,gam0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,gam0],[0,0,0,0,0,0,0,0]],
            [[0, 0,gam1,0,0,0,0,0], [0,0,0,gam1,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,gam1,0],[0,0,0,0,0,0,0,gam1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]],
            [[0, 0,0,0,gam2,0,0,0], [0,0,0,0,0,gam2,0,0], [0,0,0,0,0,0,gam2,0], [0,0,0,0,0,0,0,gam2],
             [0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
            ])
        
        if lb_type =='decay':
            # Rabi osscilation Hamiltonian + rydberg interaction
            H = om0*np.kron(np.kron(X,Id),Id) + om1*np.kron(np.kron(Id,X),Id)+om2*np.kron(np.kron(Id,Id),X) \
                + ryd_interaction*np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0],
                                            [0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,2]])
        elif lb_type == 'tfim':
            # Transverse field Ising model Hamiltonian
            H = j_en *(np.kron(np.kron(Z, Id), Id) @np.kron(np.kron(Id, Z), Id) \
                       + np.kron(np.kron(Id, Id), Z) @np.kron(np.kron(Id, Z), Id)) \
                - h_en *(np.kron(np.kron(X,Id), Id)+np.kron(np.kron(Id,X), Id) + np.kron(np.kron(Id, Id), X))
                
        else:
            print("Lindblad type {} invalid, quantum channel contains only simple decay".format(lb_type))
    
    stinespring_class.set_original_lindblad(H, An, t_lb)
else:
    # Define the Unitary circuit to imitate
    #U = np.eye(2**(2*m))
    U_init_theta = np.random.rand(depth,2*m,3)*np.pi
    #U_init_theta = np.ones([depth, 2*m, 3])
    
    circuit = U_circuit(m=2*m, circuit_type = circuit_type, **entangle_pars)
    U = circuit.gate_circuit(U_init_theta, n = repeats)
    stinespring_class.set_original_unitary(U)
    
    t_lb  = 0


#%% Set random initial rho and their evolutions under U or Lindblad

stinespring_class.set_training_data(n_training,seed, paulis = pauli_type, t_repeated = nt_training)

if from_lindblad:
    t0 = 0
    t1 = t_lb
    steps = 100
    t_range = np.linspace(t0,t1,steps)
    sol = np.zeros([steps, 2**m, 2**m])
    for i, ti in enumerate(t_range):
        stinespring_class.set_original_lindblad(H,An,ti)
        sol[i] = np.real(stinespring_class.evolution(rho0))
    
    qubit_strings = np.array(["" for _ in range(2**m)], dtype = '<U{}'.format(m))
    for i in range(2**m):
            qubit_strings[i] = format(i, '0{}b'.format(m))
         
    plt.figure()
    for i in range(2**m):
        for j in range(2**m):
            plt.plot(t_range, sol[:,i,j], label = r'$|{}\rangle \langle{}|$'.format(qubit_strings[i],qubit_strings[j]) )
    plt.legend(loc=0, bbox_to_anchor=(1.0, 1.0))
    plt.xlim([0,t_lb])
    
    plt.figure()
    for i in range(2**m):
        plt.plot(t_range, sol[:,i,i], label = r'$|{}\rangle \langle{}|$'.format(qubit_strings[i],qubit_strings[i]) )
    plt.legend()
    plt.xlim([0,t_lb])
    
    print("Evolution tested")


#%% Training of unitary

stinespring_class.set_unitary_circuit(circuit_type = circuit_type, depth = depth, gate_par = gate_par)
print("Initial error: ", stinespring_class.training_error(theta0))

if from_lindblad:
    theta1, error1 = stinespring_class.run_all_lindblad(H, An, t_lb, **train_par, **entangle_pars)
else:
    theta1, error1 = stinespring_class.run_all_unitary(U, **train_par, **entangle_pars)

theta_opt, gate_par_opt = stinespring_class.reshape_theta_phi(stinespring_class.theta_opt)

print("Unitary trained")

plt.figure()
plt.plot(error1[1:])
plt.yscale('log')
plt.ylabel('Error')
plt.xlabel('Iteration')
if save_figs:
    plt.savefig('Figures//{}.svg'.format(name), bbox_inches = 'tight')
    plt.savefig('Figures//{}.pdf'.format(name), bbox_inches = 'tight')
    
if circuit_type == 'pulse based':
    theta1 = stinespring_class.reshape_theta_phi(theta1)[0]
    plt.figure()
    for k in range(theta1.shape[0]):
        colours = ['b', 'r', 'g', 'm', 'y', 'k']
        plt.plot(np.linspace(0,T_pulse, Zdt), theta1[k,:,0], '{}-'.format(colours[k%6]), label = 'qubit {}'.format(k))
        plt.plot(np.linspace(0,T_pulse, Zdt), theta1[k,:,1], '{}--'.format(colours[k%6]))
        plt.legend()
    if save_figs:
        plt.savefig('Figures//{} Pulse.svg'.format(name), bbox_inches = 'tight')
    



#%% Reapplying unitary
#prediction_iterations = 50

# Set new rho0
stinespring_class.set_training_data(n_training, seed+3, paulis = pauli_type, t_repeated = nt_training)

# rho0 index for plotting
rho_i = 3

# Initialize empty arrays
error = np.zeros(prediction_iterations)
ev_exact = np.zeros([n_training, prediction_iterations, 2**m, 2**m], dtype = np.complex128)
ev_exact_full = np.zeros([100, 2**m, 2**m], dtype = np.complex128)
ev_circuit = np.zeros([n_training, prediction_iterations, 2**m, 2**m], dtype = np.complex128)
trace_dist = np.zeros([n_training, prediction_iterations])

for i, rho in enumerate(stinespring_class.training_data[0]):
    ev_exact[i] = stinespring_class.evolution_n(prediction_iterations, rho)[1:]
    ev_circuit[i] = stinespring_class.unitary_approx_n(prediction_iterations, rho)[1:]
    
    for n in range(prediction_iterations):
        ev_exact_root = sc.linalg.sqrtm(ev_exact[i,n])
        
        trace_dist[i,n] = max(0, 1 -np.abs(np.trace(sc.linalg.sqrtm(ev_exact_root @ ev_circuit[i,n] @ ev_exact_root))))

error = np.einsum('ij->j', trace_dist)/n_training

ev_exact_full = np.real(stinespring_class.evolution_t(np.linspace(0,prediction_iterations*t_lb,200), stinespring_class.training_data[0,rho_i]))

colours = ['b', 'r', 'g', 'm', 'y', 'k']
plt.figure()
x_exact = np.linspace(0, prediction_iterations*t_lb,200)
x_approx = np.array(range(1, (prediction_iterations+1)))*t_lb
for i in range(2**m):
    plt.plot(x_exact, ev_exact_full[:,i,i], '{}-'.format(colours[i%6]), label = r'$|{}\rangle \langle{}|$'.format(qubit_strings[i],qubit_strings[i]) )
    plt.plot(x_approx, np.real(ev_circuit[rho_i,:,i,i]), '{}x'.format(colours[i%6]) )
    plt.plot(np.linspace(0, prediction_iterations*t_lb,3), np.zeros(3)+np.real(stinespring_class.steady_state[i,i]), '{}--'.format(colours[i%6]))
plt.legend(loc = 'upper right')
plt.xlabel("System evolution time")
plt.ylabel("Population")
plt.xlim([0,prediction_iterations*t_lb])
plt.ylim(bottom=0)
if save_figs:
    plt.savefig('Figures//{} prediction single rho.pdf'.format(name), bbox_inches = 'tight')


plt.figure()
for i in range(2**m):
    plt.plot(range(1,prediction_iterations+1), np.real(ev_exact[rho_i,:,i,i] - ev_circuit[rho_i,:,i,i]), '{}o'.format(colours[i%6]), label = r'$|{}\rangle \langle{}|$'.format(qubit_strings[i],qubit_strings[i]) )
plt.plot(range(1,prediction_iterations+1),np.zeros(prediction_iterations), 'k--')
plt.legend()
plt.xlabel("System evolution time")
plt.ylabel("Population error")
plt.xlim([1,prediction_iterations])
if save_figs:
    plt.savefig('Figures//{} prediction single rho error.pdf'.format(name), bbox_inches = 'tight')

plt.figure()
plt.plot(range(1, prediction_iterations+1), error)
plt.xlabel("Repetitions of U")
plt.ylabel("Error")
plt.xlim([1,prediction_iterations])
if save_figs:
    plt.savefig('Figures//{} predictions.svg'.format(name), bbox_inches = 'tight')
    plt.savefig('Figures//{} predictions.pdf'.format(name), bbox_inches = 'tight')








