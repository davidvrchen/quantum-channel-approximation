# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:35:01 2023

@author: lviss
"""

import numpy as np
import qutip as qt
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.axis as pltax
plt.style.use('./Plot_styles/report_style.mplstyle')
    
from stinespring_t_update_classes import stinespring_unitary_update, U_circuit

from Stinespring_unitary_circuits import generate_gate_connections
    

#%% Initialization of parameters
save_figs = False                   # Save figures as pdf and svg
name = 'test run'                   # name to prepend to all saved figures

# General parameters
m = 1
n_training = 10                    # Number of initial rho's to check, last one is steady state
nt_training = 2                     # Number of repeated timesteps per rho
prediction_iterations = 20          # Number of reaplications of the found unitary to check for evolution of errors
seed = 5                            # Seed for random initial rho's
error_type = 'pauli trace'          # Type of error: "measurement n", "pauli trace", "bures", "trace", 'wasserstein', 'trace product' 
steadystate_weight = 0              # Weight given to steady state density matrix in calculation of error
pauli_type = 'full'              # Pauli spin matrices to take into account. 
                                    # Options: 'full', 'order k' for k-local, 'random n'
                                    
circuit_type = 'ryd'            # Gate type used to entangle, 
                                    #   choose: cnot, ryd, xy, decay, with varied parameters
                                    # choose: 'pulse based'
qubit_structure = 'triangle d = 0.90'        # structure of qubits: pairs, loose_pairs, triangle, line
                                    # add d = some number to scale the distance between all qubits

# Gate based circuit parameters
cutoff = False                       # Cutoff interactions above distance 1 for gate based circuit
depth = 10                           # Depth of simulation circuit (depth-1 entanglement gates)
repeats = 2                         # Number of identical circuits (depth-1), with applying exp(itH)
n_grad_directions = 25              # Number of parameters to calculate the gradient for simultaneous 
                                    # (for stochastic gradient descend), set to -1 for full gradient

phi = np.pi/10                      # Initial phi guess (for xy and xy_var)
t_ryd = 1.0                           # Interaction time for the rydberg entangle gate
gammat = 0.1                        # Decay rate for decay entangle gate


# Pulse based parameters
T_pulse = 20                         # Pulse duration 
driving_H_interaction = 'rydberg11'   # basic11, rydberg11, dipole0110
control_H = 'realrotations+11'             # Control Hamiltonian ('rotations' or 'realrotations', +11 for detuning)
lambdapar = 10**(-4)                # Weight on L2 norm of pulse
Zdt = 101


# Armijo gradient descend parameters
max_it_training = 50    # Max number of Armijo steps in the gradient descend
sigmastart = 10          # Starting sigma
gamma = 10**(-4)        # Armijo update criterion
epsilon = 10**(-4)      # Finite difference stepsize for gate based gradient

# Quantum channel to approximate defined by a Lindbladian or by another unitary circuit
from_lindblad = True

# Lindblad equation parameters
lb_type = 'decay' # Type of quantum channel to approx, 
                    # 'decay' is decay, rabi oscillations per qubit and rydberg interaction
                    # 'tfim' is transverse field ising model with decay
t_lb = 0.5       # Evolution time steps
gam0 = 0.5     # Decay rate qubit 1
gam1 = 0.3      # Decay rate qubit 2
gam2 = 0.2      # Decay rate qubit 3
gam3 = 0.1      #

#type 'decay', rabi oscillations:
om0 = 0.3         # Rabi oscillation frequency qubit 1
om1 = 0.5        # Hamiltonian forcing strength qubit 2
om2 = 0.35      # Hamiltonian forcing strength qubit 3
ryd_interaction = 0.2 # 0.2 #Rydberg interaction strength between the qubits

#tfim:
j_en = 1    # neighbour-neighbour coupling strength for transverse field ising model
h_en = 1    # Transverse magnetic field strength

#### Set parameter dependent things ####

# Seed randomisers
np.random.seed(seed)

# rho0, used for plotting the evolution of the Lindblad equation (if used)
# Various options:
    
# =============================================================================
# # |11><11| state
# rho0 = np.zeros([2**m,2**m])
# rho0[3,3] = 1
# =============================================================================

# =============================================================================
# # Fully mixed
# rho0 = np.eye(2**m)/2**m
# =============================================================================

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
    
try:
    if stinespring_class.theta_opt.size == theta0.size:
        theta0 = stinespring_class.theta_opt
        print("Start with previous theta")
    else:
        print("Start with random theta, sizes do not match")
except NameError:
    print("Start with random theta, no theta_opt found")
    

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
        H = om0*X
    elif m==2:
        An = np.array([[[0,gam0**(1/2),0,0],[0,0,0,0],[0,0,0,gam0**(1/2)],[0,0,0,0]], #|. 1> to |. 0>
                      [[0,0,gam1**(1/2),0],[0,0,0,gam1**(1/2)],[0,0,0,0],[0,0,0,0]], #|1 .> to |0 .>
                      ])
        
# =============================================================================
#         An = np.array([[[0,gam0**(1/2),0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], #|01> to |00>
#                        [[0,0,0,0],[0,0,0,0],[0,0,0,gam1**(1/2)],[0,0,0,0]], #|11> to |10>
#                        [[0,0,gam2**(1/2),0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], #|10> to |00>
#                        [[0,0,0,0],[0,0,0,gam3**(1/2)],[0,0,0,0],[0,0,0,0]]  #|11> to |01>
#                       ])
# =============================================================================
        
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

#%% Reapplying unitary
#prediction_iterations = 50

# Set new rho0
stinespring_class.set_training_data(n_training, seed+2, paulis = pauli_type, t_repeated = nt_training)

# rho0 index for plotting
rho_i = 0

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

#%% Making plots

name = name
save_figs = save_figs
colours = ['b', 'r', 'g', 'darkorchid', 'gold', 'k']

# Error over iterations
fig = plt.figure()
ax = plt.subplot(111)
plt.plot(range(1, len(error1)), error1[1:], color = 'gray', linewidth = 3)
plt.yscale('log')
plt.ylabel(r'$J_1(U)$ ')
plt.xlabel('Iteration')
plt.xlim(left = 0, right = max_it_training)
if save_figs:
    #plt.savefig('Figures//{}.svg'.format(name), bbox_inches = 'tight')
    plt.savefig('Figures//{} - training error.pdf'.format(name), bbox_inches = 'tight')

ax.text(-0.18, 0.95, '(b)', transform=ax.transAxes, fontsize = 16)

if m ==1:
    geom_x = [0.77,0.72,0.82]
    geom_y = [0.30,0.18,0.18]
    pairs = [[0,1], [0,2], [1,2]]
    dot_colours = ['b', 'r', 'r']
elif m ==2:
    geom_x = [0.67,0.77,0.62,0.72,0.82]
    geom_y = [0.30,0.30, 0.18,0.18,0.18]
    pairs = [[0,1], [0,2], [0,3],[1,3],[1,4],[2,3],[3,4]]
    dot_colours = colours

for pair1, pair2 in pairs:
    plt.plot([geom_x[pair1], geom_x[pair2]],[geom_y[pair1],geom_y[pair2]], c='gray', transform = ax.transAxes)

for k in range(2*m+1):
    plt.scatter(geom_x,geom_y, color = dot_colours[0:2*m+1], transform = ax.transAxes, zorder = 2)
    
ax.text(0.82, 0.24, r'$R = 0.85$', transform=ax.transAxes, fontsize = 12)
    
subplot = plt.axes([0.55, 0.5, 0.3, 0.3])
# Final pulses
if circuit_type == 'pulse based':
    theta1 = stinespring_class.reshape_theta_phi(theta1)[0]
    #plt.figure()
    legend_elements = []
    #legend_elements.append(Line2D([0],[0], color = 'k', ls = '-', lw = 2, label = 'Armijo descend'))
    
    legend_elements.append(Line2D([0],[0], color = colours[0], ls = '-', lw = 2, label = r'$q_{0}$'))
    legend_elements.append(Line2D([0],[0], color = colours[1], ls = '-', lw = 2, label = r'$q_{1}$ & $q_{2}$'))
    
    x_range = np.linspace(0,stinespring_class.T_pulse, stinespring_class.Zdt)
    for k in range(2):
        
        
# =============================================================================
#         # real and imaginary
#         plt.plot(x_range, theta1[k,:,0], '-', color = colours[k%6], label = 'qubit {}'.format(k))
#         plt.plot(x_range, theta1[k,:,1], ':', color = colours[k%6])
#         legend_elements.append(Line2D([0],[0], color = colours[k%6], ls = '-', lw = 2, label = 'qubit {}'.format(k)))
#         if stinespring_class.control_H.shape[0] == 2*(2*m+1):
#             plt.plot(x_range, theta1[2*stinespring_class.m+1+k,:,0], '--', color = colours[k%6])
#             plt.plot(x_range, theta1[2*stinespring_class.m+1+k,:,1], '-.', color = colours[k%6])
# =============================================================================
            
        # Real only
        plt.plot(x_range, theta1[k,:,0]+theta1[k,:,1], color = colours[k%6], label = 'q {}'.format(k))
        #legend_elements.append(Line2D([0],[0], color = colours[k%6], ls = '-', lw = 2, label = r'$q_{a}$'.format(a=k)))
        if stinespring_class.control_H.shape[0] == 2*(2*m+1):
            plt.plot(x_range, theta1[2*stinespring_class.m+1+k,:,0], '--', color = colours[k%6])
            
    
    if stinespring_class.control_H.shape[0] == 2*(2*m+1):
# =============================================================================
#         legend_elements.append(Line2D([0],[0], color = 'gray', ls = '-', lw = 2, label = 'coupling - real'))
#         legend_elements.append(Line2D([0],[0], color = 'gray', ls = ':', lw = 2, label = 'coupling - imaginary'))
#         legend_elements.append(Line2D([0],[0], color = 'gray', ls = '--', lw = 2, label = 'detuning - real'))
#         legend_elements.append(Line2D([0],[0], color = 'gray', ls = '-.', lw = 2, label = 'detuning - imaginary'))
# =============================================================================
        
        legend_elements.append(Line2D([0],[0], color = 'k', ls = '-', lw = 2, label = 'coup'))
        legend_elements.append(Line2D([0],[0], color = 'k', ls = '--', lw = 2, label = 'det'))
    else:
        legend_elements.append(Line2D([0],[0], color = 'gray', ls = '-', lw = 2, label = 'real'))
        legend_elements.append(Line2D([0],[0], color = 'gray', ls = ':', lw = 2, label = 'imaginary'))
    #plt.legend(handles = legend_elements, loc = (-0.9,-0.2))
    plt.legend(handles = legend_elements, loc = (-1.0, 0.3))
    
    plt.xlabel(r'$\tau$ [ms]')
    plt.xlim([0,stinespring_class.T_pulse])
    plt.ylabel(r'$z_r$ [kHz]')
    plt.title("Final pulses")
    
    if save_figs:
        plt.savefig('Figures//{} training and pulse.pdf'.format(name), bbox_inches = 'tight')


# Prediction on a single rho
colours = ['b', 'r', 'g', 'm', 'y', 'k']
fig = plt.figure()
ax = plt.subplot(111)
x_exact = np.linspace(0, prediction_iterations*t_lb,200)
x_approx = np.array(range(1, (prediction_iterations+1)))*t_lb
legend_elements = []

legend_elements.append(Line2D([0],[0], color = 'k', ls = '-', lw = 2, label = 'exact'))
legend_elements.append(Line2D([0],[0], color = 'k', ls = '--', lw = 2, label = 'steady state'))
legend_elements.append(Line2D([0],[0], color = 'k', marker = 'x', lw = 0, label = 'approximation'))

for i in range(2**m):
    plt.plot(x_exact, ev_exact_full[:,i,i], '{}-'.format(colours[i%6]))
    plt.plot(x_approx, np.real(ev_circuit[rho_i,:,i,i]), '{}x'.format(colours[i%6]) )
    plt.plot(np.linspace(0, prediction_iterations*t_lb,3), np.zeros(3)+np.real(stinespring_class.steady_state[i,i]), '{}--'.format(colours[i%6]))
    legend_elements.append(Line2D([0],[0], color = colours[i%6], ls = '-', lw = 2, label = r'$|{0}\rangle \langle{0}|$'.format(qubit_strings[i])))


plt.ticklabel_format(axis='both', style='sci', scilimits=(-3,3))

#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#plt.legend(handles = legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(handles = legend_elements, loc = 'upper right', ncols=2)

plt.xlabel("t [a.u.]")
plt.ylabel("Population")
plt.xlim([0,prediction_iterations*t_lb])

plt.axvline(x=nt_training*t_lb, ls = ':', color = 'dimgray', linewidth = 2)
mayor_tx = list(range(0,11,2))#+[nt_training*t_lb]
mayor_tx_lab = list(map(str, mayor_tx))
if int(nt_training*t_lb) in mayor_tx:
    tick_index = mayor_tx.index(int(nt_training*t_lb))
    mayor_tx_lab[tick_index] = r'$t_{train}$'
else:
    mayor_tx = mayor_tx + [nt_training*t_lb]
    mayor_tx_lab = mayor_tx_lab + [r'$t_{train}$']
    tick_index = -1
minor_tx = list(np.arange(0,10,0.5))
plt.xticks(ticks = mayor_tx, labels = mayor_tx_lab)
plt.xticks(ticks = minor_tx, minor = True)
#plt.xticks(ticks = mayor_tx,labels = ['0', r'$t_{train}$', '2', '4', '6', '8', '10'])
#plt.xticks(ticks = [0.5,1.5,2.5,3,3.5,4.5,5,5.5,6.5,7,7.5,8.5,9,9.5],minor = True)

ax.get_xticklabels()[tick_index].set_color("dimgray")

#plt.ylim(bottom=0)
if save_figs:
    plt.savefig('Figures//{} single rho.pdf'.format(name), bbox_inches = 'tight')

ax.text(-0.15, 0.95, '(a)', transform=ax.transAxes, fontsize = 16)

# Error on prediction of a single rho
#plt.figure()
#subplot = plt.axes([0.55, 0.65, 0.3, 0.2])
subplot = plt.axes([0.55, 0.2, 0.3, 0.2])
for i in range(2**m):
    plt.plot(x_approx, np.real(ev_exact[rho_i,:,i,i] - ev_circuit[rho_i,:,i,i]), '{}x'.format(colours[i%6]), label = r'$|{}\rangle \langle{}|$'.format(qubit_strings[i],qubit_strings[i]) )
    
# =============================================================================
#     plt.plot(x_approx, np.real(ev_exact[rho_i,:,i,i] - evolution_base[rho_i,:,i,i]), '{}x'.format(colours[i%6]), label = r'$|{}\rangle \langle{}|$'.format(qubit_strings[i],qubit_strings[i]) )
#     plt.plot(x_approx, np.real(ev_exact[rho_i,:,i,i] - evolution_detuning[rho_i,:,i,i]), '{}o'.format(colours[i%6]), label = r'$|{}\rangle \langle{}|$'.format(qubit_strings[i],qubit_strings[i]) )
#     plt.plot(x_approx, np.real(ev_exact[rho_i,:,i,i] - evolution_steadystate[rho_i,:,i,i]), '{}v'.format(colours[i%6]), label = r'$|{}\rangle \langle{}|$'.format(qubit_strings[i],qubit_strings[i]) )
#     plt.plot(x_approx, np.real(ev_exact[rho_i,:,i,i] - evolution_multitime[rho_i,:,i,i]), '{}^'.format(colours[i%6]), label = r'$|{}\rangle \langle{}|$'.format(qubit_strings[i],qubit_strings[i]) )
# 
# legend_elements = [Line2D([0],[0], color = 'b', lw = 2, label = r'$|0\rangle \langle 0 |$'),
#                    Line2D([0],[0], color = 'r', lw = 2, label = r'$|1\rangle \langle 1 |$'),
#                    Line2D([0],[0], color = 'gray', lw = 0, marker = 'x', label = 'base'),
#                    Line2D([0],[0], color = 'gray', lw = 0, marker = 'o', label = 'detuning'),
#                    Line2D([0],[0], color = 'gray', lw = 0, marker = 'v', label = 'steadystate'),
#                    Line2D([0],[0], color = 'gray', lw = 0, marker = '^', label = 'multitime')]
# plt.legend(handles = legend_elements)
# =============================================================================
plt.plot(x_exact,np.zeros(200), color = 'dimgray', ls = ':')
#plt.xlabel("System evolution time")
plt.ylabel("Error")
plt.ticklabel_format(axis='both', style='sci', scilimits=(-3,3))
plt.xlim([0,prediction_iterations*t_lb])
if save_figs:
    plt.savefig('Figures//{} prediction single rho and error.pdf'.format(name), bbox_inches = 'tight')
    

# Evolution of error over prediction time
plt.figure()
plt.plot(np.linspace(1*t_lb, prediction_iterations*t_lb,len(error)), error)
plt.xlabel("System evolution time")
plt.ylabel("Error - Bures")
plt.xlim([0,prediction_iterations*t_lb])
plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,3))
if save_figs:
    plt.savefig('Figures//{} predictions total error.svg'.format(name), bbox_inches = 'tight')
    plt.savefig('Figures//{} predictions total error.pdf'.format(name), bbox_inches = 'tight')


#%% Plots for comparisons (eg gate vs stochastic gate vs pulse)

plt.figure()
maxshots = 400
plt.plot(np.linspace(0,maxshots,len(error1_gate)-1), error1_gate[1:], label = 'gate')
plt.plot(np.linspace(0,maxshots,len(error1_gate_stoch)-1), error1_gate_stoch[1:], label = 'gate stoch')
plt.plot(np.linspace(0,maxshots,len(error1_pulse)-1), error1_pulse[1:], label = 'pulse')
# =============================================================================
# plt.plot(np.linspace(0,maxshots,len(error1_base)-1), error1_base[1:], label = 'Base')
# plt.plot(np.linspace(0,126,125), error1_det[1:126], label = 'Detuning')
# plt.plot(np.linspace(0,maxshots,len(error1_mult)-1), error1_mult[1:], label = '2 time steps')
# plt.plot(np.linspace(0,maxshots,len(error1_ss)-1), error1_ss[1:], label = 'Steady state')
# =============================================================================
plt.legend()
plt.yscale('log')
plt.ylabel('Error - {}'.format(error_type))
plt.xlabel('Fraction of total shots')
#plt.xlabel("Armijo gradient descend steps")
plt.xlim(left = 0, right = maxshots)
if save_figs:
    #plt.savefig('Figures//{}.svg'.format(name), bbox_inches = 'tight')
    plt.savefig('Figures//{} - training error comparison.pdf'.format(name), bbox_inches = 'tight')

plt.figure()
plt.scatter(x_approx, np.abs(np.real(ev_exact[rho_i,:,0,0] - ev_circuit_gate[rho_i,:,0,0])), label = 'gate')
plt.scatter(x_approx, np.abs(np.real(ev_exact[rho_i,:,0,0] - ev_circuit_gate_stoch[rho_i,:,0,0])), label = 'gate stoch')
plt.scatter(x_approx, np.abs(np.real(ev_exact[rho_i,:,0,0] - ev_circuit_pulse[rho_i,:,0,0])), label = 'pulse')

# =============================================================================
# plt.scatter(x_approx, np.abs(np.real(ev_exact[rho_i,:,0,0] - ev_circuit_base[rho_i,:,0,0])), label = 'Base')
# plt.scatter(x_approx, np.abs(np.real(ev_exact[rho_i,:,0,0] - ev_circuit_det[rho_i,:,0,0])), label = 'Detuning')
# plt.scatter(x_approx, np.abs(np.real(ev_exact[rho_i,:,0,0] - ev_circuit_mult[rho_i,:,0,0])), label = '2 time steps')
# plt.scatter(x_approx, np.abs(np.real(ev_exact[rho_i,:,0,0] - ev_circuit_ss[rho_i,:,0,0])), label = 'Steady state')
# =============================================================================

plt.legend()
plt.yscale('log')
#plt.plot(x_exact,np.zeros(200), 'k--')
plt.xlabel("System evolution time")
plt.ylabel("Population error")
#plt.ticklabel_format(axis='both', style='sci', scilimits=(-3,3))
plt.xlim([0,prediction_iterations*t_lb])
if save_figs:
    plt.savefig('Figures//{} prediction single rho error.pdf'.format(name), bbox_inches = 'tight')

plt.figure()
plt.scatter(np.linspace(1*t_lb, prediction_iterations*t_lb,len(error)), error_gate, label = 'gate')
plt.scatter(np.linspace(1*t_lb, prediction_iterations*t_lb,len(error)), error_gate_stoch, label = 'gate stoch')
plt.scatter(np.linspace(1*t_lb, prediction_iterations*t_lb,len(error)), error_pulse, label = 'pulse')
# =============================================================================
# plt.scatter(np.linspace(1*t_lb, prediction_iterations*t_lb,len(error)), error_base, label = 'Base')
# plt.scatter(np.linspace(1*t_lb, prediction_iterations*t_lb,len(error)), error_det, label = 'Detuning')
# plt.scatter(np.linspace(1*t_lb, prediction_iterations*t_lb,len(error)), error_mult, label = '2 time steps')
# plt.scatter(np.linspace(1*t_lb, prediction_iterations*t_lb,len(error)), error_ss, label = 'Steady state')
# =============================================================================

plt.legend()
plt.yscale('log')
plt.xlabel("System evolution time")
plt.ylabel("Bures Error on predictions")
plt.xlim([0,prediction_iterations*t_lb])
#plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,3))
if save_figs:
    #plt.savefig('Figures//{} predictions total error.svg'.format(name), bbox_inches = 'tight')
    plt.savefig('Figures//{} predictions total error.pdf'.format(name), bbox_inches = 'tight')






