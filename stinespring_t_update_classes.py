# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:41:00 2023

@author: lviss
"""
import numpy as np
import qutip as qt
import scipy as sc
import torch as to
import random as rd
import math
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animate
from numpy.core.umath_tests import inner1d
import time

from my_functions import get_paulis, create_driving_hamiltonians, create_control_hamiltonians, \
    Znorm, wasserstein1
from Stinespring_unitary_circuits import U_circuit, U_circuit_pulse, generate_gate_connections


class stinespring_unitary_update:

    def __init__(self, m=2, error_type = 'density rho', circuit_type = 'xy', par_dict = {}):
        """
        Settings for quantum simulations with the kernel trick

        Parameters
        ----------
        m : int, optional
            Number of qubits in the system (without support qubits). The default is 2.
        error_type : str, optional
            Type of error to use:
                - 'density rho' for density matrices (default)
                - 'trace' for traces with pauli spin matrices
                - 'measurement' for actual measurements (binomial from trace)
        circuit_type : str, optional
            Type of circuit to use:
                - 'pulse based' for pulse based
                - 'xy' for gate based on parameterized swap gate
                - 'decay' for gate based on analytical decay
                - 'cnot' for gate based on cnot gate
                - 'rydberg' for gate based on rydberg interaction gate
        par_dict : dictionary
            Dictionary with additional parameters
            General:
                - 'qubit structure' : 
                    - pairs
                    - loose_pairs
                    - triangle
                    - line
            Gate based:
                - 'n_grad_directions' : -1 for all directions, otherwise stochastic gd with n directions
                - cutoff : if True, no gates between qubits further away than distance 1
            Pulse based:
                - T_pulse : Total pulse duration
                - control_H : name of control Hamiltonians to use
                - driving_H_interaction : name of driving Hamiltonians to use
                - lambdapar : lambda parameter for armijo gradient descend
                - Zdt : Number of steps in discretization of the pulse

        Returns
        -------
        None.

        """
        
        #System settings
        self.m = m
        state_00 = np.zeros([2**(self.m+1), 2**(self.m+1)])
        state_00[0,0] = 1
        self.state_00 = state_00
        
        self.weights = 0
        self.steadystate_weight = par_dict['steadystate_weight']
        
        # Set up error variables
        if 'measurement' in error_type:
            self.error_type = 'measurement'
            self.n_measurements = int(re.findall('\d+\Z', error_type)[0])
        else:
            self.error_type = error_type
            
        # Set up circuit variables
        self.circuit_type = circuit_type
        self.qubit_structure = par_dict['qubit_structure']
        
        if circuit_type == 'pulse based':
            self.T_pulse = par_dict['T_pulse']
            self.control_H = create_control_hamiltonians(2*m+1,par_dict['control_H'])
            self.n_controls = self.control_H.shape[0]
            self.driving_H = create_driving_hamiltonians(2*m+1,par_dict['driving_H_interaction'],par_dict['qubit_structure'])
            self.lambdapar = par_dict['lambdapar']
            self.Zdt = par_dict['Zdt']
        else:
            self.n_grad_directions = par_dict['n_grad_directions']
            self.num_gate_pairs = generate_gate_connections(self.m, self.qubit_structure, cutoff = par_dict['cutoff'])
            
        # Set up time variables
        self.time_circuit = 0

    
    def update_pars(self,**kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
            self.__dict__[key] = kwargs[key]
            
    def time_wrapper(func):
        """
        Decorator to time class methods.
        Modify the function to time the total function evaluation time and
        count the number of method calls. Data is saved as class paramater as
        'timed_{}' and 'calls_{}' with {} = function name

        Parameters
        ----------
        func : method of class
            

        Returns
        -------
        func : decorated func

        """
        def innerwrap(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            
            try:
                args[0].__dict__['timed_' + str(func.__name__)] += end-start
            except KeyError:
                args[0].__dict__['timed_' + str(func.__name__)] = end-start
                
            try:
                args[0].__dict__['calls_' + str(func.__name__)] += 1
            except KeyError:
                args[0].__dict__['calls_' + str(func.__name__)] = 1
            
            return result
        return innerwrap
    
    def print_times(self):
        """
        Prints the times and calls recorded by the time_wrapper

        Returns
        -------
        None.

        """
        print('-----')
        for key, val in self.__dict__.items():
            if key.startswith('timed_'):
                print("Function", key[6:], "took {0:d}:{1:.2f} min:sec".format(int(val//60), val%60), end = '')
                print(" for {} calls".format(self.__dict__['calls_'+key[6:]]))
        print('-----')
            
    def run_all_lindblad(self, H, An, t, n_training, seed, depth, 
                         theta0, max_it_training, epsilon = 0.01, 
                         gamma = 10**(-4), sigmastart=1, circuit_type = 'ryd', 
                         pauli_type = 'order 1', t_repeated = 2, **kwargs):
        self.depth = depth

        self.set_original_lindblad(H, An, t)
        
        self.set_training_data(n_training,seed, paulis= pauli_type, t_repeated = t_repeated)
        
        self.set_unitary_circuit(depth = depth, circuit_type =circuit_type, **kwargs)
        
        self.run_armijo(theta0, max_it_training, gamma = gamma, 
                        sigmastart=sigmastart, epsilon = epsilon)
        
        self.print_times()

        return self.theta_opt, self.error
    
    def run_all_unitary(self, U, n_training, seed, depth, 
                         theta0, max_it_training, epsilon = 0.01, 
                         gamma = 10**(-4), sigmastart=1, circuit_type = 'ryd',
                         pauli_type = 'order 1', t_repeated = 2, **kwargs):
        self.depth = depth

        self.set_original_unitary(U)
        
        self.set_training_data(n_training,seed, paulis = pauli_type, t_repeated = t_repeated)
        
        self.set_unitary_circuit(depth = depth, circuit_type =circuit_type, **kwargs)
        
        self.run_armijo(theta0, max_it_training, gamma = gamma, 
                        sigmastart=sigmastart, epsilon = epsilon)

        return self.theta_opt, self.error
        
    
    def set_original_lindblad(self, H, An, t_ham):
        """
        Sets self.evolution(rho0) as a function that simulates a lindbladian:
        i d_t rho = [H,rho] + sum_k An[k] rho An[k]* - 1/2 [An[k]*An[k], rho]

        Parameters
        ----------
        H : np.ndarray, 2**m x 2**m
            Hamiltonian acting on the qubits.
        An : np.ndarray, n by 2**m x2**m
            An matrices in the lindbladian.
        t_ham : float
            fixed time to integrate the lindbladian to.

        """
        self.H = H
        self.t_ham = t_ham
        m = self.m
        def lindblad_evolution(t_eval, rho0):
            
            def lindbladian(t,rho):
                rho = np.reshape(rho,(2**m,2**m))
                result = -1j*(H@rho - rho@H)
                for A in An:
                    Ad = np.conj(np.transpose(A))
                    result = result + A @rho @Ad - Ad @A @rho/2 - rho @Ad @A/2
                result = np.reshape(result,4**m)
                return result
            
            solver = sc.integrate.complex_ode(lindbladian)
            solver.set_initial_value(np.reshape(rho0,4**m),0)
            
            if type(t_eval) == np.ndarray:
                sol = np.zeros([len(t_eval), 4**m], dtype = np.complex128)
                sol[0] = solver.integrate(t_eval[0])
                for i in range(1,len(t_eval)):
                    solver.set_initial_value(sol[i-1],t_eval[i-1])
                    sol[i] = solver.integrate(t_eval[i])
                return np.reshape(sol, [len(t_eval), 2**m, 2**m])
                    
            else:
                sol = solver.integrate(t_eval)
                if solver.successful():
                    return np.reshape(sol,[2**m,2**m])
                else:
                    print("Solver for lindblad evolution aborted")
                    return rho0
            
        self.evolution = lambda rho0: lindblad_evolution(t_ham, rho0)
        self.evolution_t = lindblad_evolution
        self.from_lindblad = True
    
    def set_original_unitary(self, U):
        """
        Sets self.evolution(rho0) as a function that simulates another unitary circuit:
        Phi_t(rho) = Tr_B[U (rho0 x |0><0|) U*]

        Parameters
        ----------
        U : np.ndarray 2**(2*m) x 2**(2*m)
            The unitary matrix on the system with twice the number of qubits.

        """
        
        def unitary_evolution(rho):
            m = self.m
            UrhoU = U @ np.kron(rho, self.state_00) @ np.transpose(np.conjugate(U))
            return np.trace(UrhoU.reshape(2**m,2**(m+1),2**m,2**(m+1)), axis1=1, axis2=3)
        self.evolution = unitary_evolution
        self.from_lindblad = False
        
    
    def set_unitary_circuit(self, depth, circuit_type = 'ryd', repeat = 1, **kwargs):
        """
        Initialises self.circuit as a U_circuit class object that can run
        self.circuit.gate_circuit(theta, **gate_parameters)

        Parameters
        ----------
        depth : int
            depth of the circuit.
        circuit_type : str, optional
            Type of entanglement gate to use for gate based simulations. options:
                - 'ryd' rydberg entanglement between all qubits (default)
                - 'decay' analytical solution to 1 qubit decay
                - 'xy' dipole-dipole interaction between system and support qubits
                - 'cnot' CNOT gate between system and support qubits
        repeat : int, optional
            number of times to repeat the full circuit (including exp(itH)). The default is 1.
        **kwargs : dictionary
            entanglement gate arguments.

        """
        self.depth = depth
        self.circuit_type = circuit_type
        self.repeat = repeat
        if circuit_type == 'pulse based':
            self.circuit = U_circuit_pulse(m = 2*self.m+1, T= self.T_pulse, 
                                           control_H = self.control_H, driving_H = self.driving_H)
        else:
            if self.from_lindblad:
                self.circuit = U_circuit(2*self.m+1, circuit_type = circuit_type,
                                         structure = self.qubit_structure,
                                         t_ham = self.t_ham, H = self.H, **kwargs)
            else:
                self.circuit = U_circuit(2*self.m+1, circuit_type = circuit_type,
                                         structure = self.qubit_structure, **kwargs)
    
    @time_wrapper
    def set_training_data(self, n_training, seed, paulis = 'order 1', t_repeated = 2):
        """
        Initialises all the training data as class parameters

        Parameters
        ----------
        n_training : int
            Number of training samples to use.
        seed : int
            Seed for the training samples, used for reproducible data.
        paulis : str, optional
            Determines the type of pauli strings that will be used as observables.
            Options: 'order k', 'full', 'random n'
            The default is 'order 1'.
        t_repeated : int, optional
            Number of repeated timesteps to use as data. The default is 2.

        Returns
        -------
        None.

        """
        m = self.m
        
        self.set_steady_state()
        
        #rho_list = np.zeros((n_training,2**m,2**m),dtype = np.csingle)
        
        # dims = n, l, matrix
        training = np.zeros((t_repeated+1, n_training,2**m,2**m),dtype = np.csingle)
        training_root = np.zeros((t_repeated+1, n_training,2**m,2**m),dtype = np.csingle)
        
        # dims = k, matrix
        paulis, pauli_names, pauli_id_list, pauli_indices = get_paulis(m, space = paulis)
        
        # dims = n, l, k (time, data, pauli)
        traces = np.zeros((t_repeated+1, n_training, len(paulis)))
        measurements = np.zeros((t_repeated+1, n_training, len(paulis)))
        
        for l in range(n_training):
            if l ==0:
                random_ket = qt.rand_ket_haar(dims = [[2**m], [1]], seed = seed)
                random_ket.dims = [[2]*m,[2]*m]
                random_bra = random_ket.dag()
                rho = (random_ket * random_bra).full()
            elif l == n_training-1:
                rho = self.steady_state
            else:
                random_ket = qt.rand_ket_haar(dims = [[2**m], [1]])
                random_ket.dims = [[2]*m,[2]*m]
                random_bra = random_ket.dag()
                rho = (random_ket * random_bra).full()
            
            #rho_list[l,:,:] = rho
            
            training[:,l,:,:] = np.reshape(self.evolution_n(t_repeated, rho),(t_repeated+1, 2**m,2**m))
            
            for t_ind in range(t_repeated+1):
                training_root[t_ind,l,:,:] = sc.linalg.sqrtm(training[t_ind,l,:,:])
                for k in range(len(paulis)):
                    traces[t_ind,l,k] = np.real(np.trace(training[t_ind,l,:,:] @ paulis[k]))
                    if self.error_type == 'measurement':
                        prob = min(max((traces[t_ind,l,k]+1)/2, 0.0), 1.0)
                        measurements[t_ind,l,k] = np.random.binomial(self.n_measurements, prob)/self.n_measurements*2 -1
                    
        #self.training_rho = rho_list       
        self.training_data = training
        self.training_data_root = training_root
        
        self.traces = traces
        self.measurements = measurements
        
        self.paulis = paulis
        self.pauli_names = pauli_names
        self.pauli_id_list = pauli_id_list
        self.pauli_indices = pauli_indices
        

    @time_wrapper
    def training_error(self, theta_phi, weights = 0, error_type = 'internal'):
        """
        Determines the error of the circuit for a given set of parameters and a
        given error type

        Parameters
        ----------
        theta_phi : np.array, 1 dimensional
            Single list of all parameters.
        weights : float or np.array, optional
            list of weights for wasserstein error (currently not implemented).
            The default is 0, which calculates the weights internally.
        error_type : str, optional
            Type of error to use. The default is 'internal' to streamline error
            type throughout the code.

        Returns
        -------
        error : float
            Total error.

        """
        if error_type == 'internal':
            error_type = self.error_type
        
        # dims = n, l, matrix
        training = self.training_data
        rho_list = self.training_data[0,:,:,:]
        roots = self.training_data_root
        m = self.m
        t_repeats, n_training_rho = training.shape[0:2]
        t_repeats -= 1
        
        theta, gate_par = self.reshape_theta_phi(theta_phi)
        
        time0 = time.time()
        U = self.circuit.gate_circuit(theta = theta, n=self.repeat, gate_par = gate_par)
        time1 = time.time()
        self.time_circuit += time1-time0
        
        error = 0
        if error_type == 'bures':
            for i in range(n_training_rho-1):
                rhos_approx = self.unitary_approx_n(t_repeats, rho_list[i], U)
                for nt in range(1,t_repeats+1):
                    error += max(0, 1- np.abs(np.trace(sc.linalg.sqrtm(np.einsum('ij, jk, kl', roots[nt,i], rhos_approx[nt], roots[nt,i], optimize = 'greedy')))))
                    
            error = error/((n_training_rho-1)*t_repeats)
            
            steadystate_approx = self.unitary_approx_n(1, rho_list[-1], U)[1]
            error += self.steadystate_weight *max(0, 1- np.abs(np.trace(sc.linalg.sqrtm(np.einsum('ij, jk, kl', roots[1,-1], steadystate_approx, roots[1,-1], optimize = 'greedy')))))
          
        elif error_type == 'trace':
            for i in range(n_training_rho-1):
                rhos_approx = self.unitary_approx_n(t_repeats, rho_list[i], U)
                for nt in range(1,t_repeats+1):
                    error += np.vdot((rhos_approx[nt] -training[nt,i]).T, rhos_approx[nt] -training[nt,i])**(1/2) /2
            
            error = np.real(error)/((n_training_rho-1)*t_repeats)
            
            steadystate_approx = self.unitary_approx_n(1, rho_list[-1], U)[1]
            error += self.steadystate_weight *np.vdot((steadystate_approx -training[1,-1]).T, steadystate_approx -training[1,-1])**(1/2) /2 
        
        elif error_type == 'pauli trace':
            rhos_approx = np.zeros((t_repeats, n_training_rho-1, 2**m, 2**m), dtype = np.csingle)
            for i in range(n_training_rho-1):
                rhos_approx[:,i] = self.unitary_approx_n(t_repeats, rho_list[i], U)[1:]
                
            # Old method, better at using all cores, but slower overall
            #pauli_rho = np.real(np.einsum('nlab, kba -> nlk', rhos_approx, self.paulis, optimize = 'greedy'))
            
            pauli_rho = np.sum(np.real(rhos_approx[:,:,self.pauli_indices[1],self.pauli_indices[0]]*self.pauli_indices[2]),axis = -1)
            
            #print(pauli_rho - pauli_rho1)
            error = (self.traces[1:,0:-1] - pauli_rho)**2
            error = np.einsum('nlk ->', error, optimize = 'greedy')/(2*n_training_rho*len(self.paulis)*t_repeats)
            error = max(0,np.real(error))
            
            steadystate_approx = self.unitary_approx_n(1, rho_list[-1], U)[1]
            pauli_rho = np.real(np.einsum('nm, kmn -> k', steadystate_approx, self.paulis, optimize = 'greedy'))
            error_add = (self.traces[1,-1,:] - pauli_rho)**2
            error_add = np.einsum('k->',error_add)/(2*n_training_rho*len(self.paulis)*t_repeats)
            #print(error, error_add)
            error += self.steadystate_weight *max(0, np.real(error_add))
            
        elif error_type == 'trace product':
            rhos_approx = np.zeros((t_repeats, n_training_rho-1, 2**m, 2**m), dtype = np.csingle)
            for i in range(n_training_rho-1):
                rhos_approx[:,i] = self.unitary_approx_n(t_repeats, rho_list[i], U)[1:]
            error = -np.einsum('nlk, kij, nlji ->', self.traces[1:,0:-1], self.paulis, rhos_approx)
            error = error/(n_training_rho*len(self.paulis)*t_repeats)
            error = np.real(error)
            
            
            steadystate_approx = self.unitary_approx_n(1, rho_list[-1], U)[1]
            error_add = - np.einsum('k, kij, ji-> ', self.traces[1,-1], self.paulis, steadystate_approx)

            error += np.real(error_add)/(n_training_rho*len(self.paulis)*t_repeats)
            
        elif error_type == 'measurement':
            for l in range(n_training_rho-1):
                rhos_approx = self.unitary_approx_n(t_repeats, rho_list[i], U)
                for nt in range(1,t_repeats+1):
                    for k in range(len(self.paulis)):
                        trace = np.real(np.trace(rhos_approx[nt] @ self.paulis[k]))
                        p = max(min((trace+1)/2,1),0)
                        measurement = np.random.binomial(self.n_measurements, p)/self.n_measurements*2 -1
                        error += (self.measurements[nt,l,k] - measurement)**2
            error = error/((n_training_rho-1)*len(self.paulis)*t_repeats)
            error = max(0, error)
            
            steadystate_approx = self.unitary_approx_n(1, rho_list[-1], U)[1]
            error_add = 0
            for k in range(len(self.paulis)):
                trace = np.real(np.trace(steadystate_approx @ self.paulis[k]))
                p = max(min((trace+1)/2,1),0)
                measurement = np.random.binomial(self.n_measurements, p)/self.n_measurements*2 -1
                error_add += (self.measurements[1,-1,k] - measurement)**2
            error += error_add/len(self.paulis)
            

        elif error_type == 'wasserstein':
# =============================================================================
#             calc_weights = False
#             if type(weights)!= np.ndarray:
#                 weights = np.zeros([t_repeats,n_training_rho, len(self.paulis)])
#                 calc_weights = True
#                 
#             for i in range(n_training_rho):
#                 rhos_approx = self.unitary_approx_n(n_training_rho, rho_list[i], U)
#                 
#                 for nt in range(1,t_repeats+1):
#                     if calc_weights:
#                         _, weights[nt-1,i,:] = wasserstein1(rhos_approx[nt], self.training_data[nt,i], (self.paulis, self.pauli_id_list))
#                         
#                     for j in range(len(self.paulis)):
#                         #error += np.trace(weights[nt,i,j]* self.paulis[j] @ (rhos_approx[nt] - self.training_data[nt,i]))     
#                         error += weights[nt-1,i,j]*np.sum(inner1d(self.paulis[j].T, rhos_approx[nt] - self.training_data[nt,i]))
#             error = np.real(error/(n_training_rho))
#             if calc_weights:
#                 self.weights = weights
# =============================================================================
                
            calc_weights = False
            if type(weights)!= np.ndarray:
                weights = np.zeros([len(self.paulis),])
                calc_weights = True
                
            rhos_approx = np.zeros((t_repeats, n_training_rho-1, 2**m, 2**m), dtype = np.csingle)
            for i in range(n_training_rho-1):
                rhos_approx[:,i] = self.unitary_approx_n(t_repeats, rho_list[i], U)[1:]
            
            rhos_approx_sum = np.sum(rhos_approx, axis = (0,1))
            rhos_exact_sum = np.sum(training[1:], axis = (0,1))
             
            if calc_weights:
                _, weights = wasserstein1(rhos_approx_sum, rhos_exact_sum, (self.paulis, self.pauli_id_list))
                self.weights = weights
            
            error = np.einsum('k, kab, ba ->', weights, self.paulis, rhos_approx_sum - rhos_exact_sum)
            error = np.real(error)
                
        elif error_type == 'rel entropy':
            pass
        
        else:
            print("Error type {} not found".format(self.error_type))
            
        if self.circuit_type == 'pulse based':
            error = error + self.lambdapar*Znorm(np.reshape(theta_phi, (self.n_controls, self.Zdt, 2)),self.T_pulse)**2
            
        return error
    
    @time_wrapper
    def find_gradient(self, theta, eps = 0.01):
        """
        Calculates the gradient for a given set of theta

        Parameters
        ----------
        theta : np.array
            for pulse based:
                - dims n_controls x Zdt x 2.
            for gate based:
                - 1 dimensional, length dependent on gate parameters
        eps : float, optional
            parameter change for finite difference for gate based.
            The default is 0.01.

        Returns
        -------
        gradient

        """
        
        
        if self.circuit_type == 'pulse based':
            num_controls=self.control_H.shape[0]
            m = self.m
            num_t, num_l = self.training_data.shape[0:2]
            num_t = num_t - 1
            num_k = len(self.paulis)
            
            gradient=np.zeros([num_controls,len(theta[0,:,0]),2])
            eta_t_sum = np.zeros([2**(2*self.m+1),2**(2*self.m+1)], dtype = np.complex128)
            
            U_full = self.circuit.full_evolution(theta)
            U = U_full[-1].full()
            Udag = np.conjugate(np.transpose(U))
            
            # Calculate all rho approximations
            rhos_approx = np.zeros((num_t+1, num_l, 2**m, 2**m), dtype = np.csingle)
            for i in range(num_l):
                rhos_approx[:,i] = self.unitary_approx_n(num_t, self.training_data[0,i], U)
            
            # Calculate all unitary transformations of the pauli matrices
            paulis_U = np.zeros((num_t, num_k, 2**m, 2**m), dtype = np.csingle)
            paulis_U[0] = self.paulis
            state_I00 = np.kron(np.eye(2**m), self.state_00)
            for i in range(1,num_t):
                pauli_ext = np.kron(paulis_U[i-1], np.eye(2**(m+1)))
                pauli_ext = np.einsum('ij, ajk, kl, lm -> aim', Udag, pauli_ext, U, state_I00, optimize='greedy')
                pauli_ext = np.trace(pauli_ext.reshape(num_k, 2**m, 2**(m+1), 2**m, 2**(m+1)), axis1=2, axis2 = 4)
                paulis_U[i] = pauli_ext
            
            # Calculate multiplicative factor of traces
            if self.error_type == 'pauli trace':
                # array with num_t (iteration), num_l (rho index), num_k (pauli index)
                # for tr[sigma (rho - rho')]^2
                traces = np.einsum('kij, nlji -> nlk', self.paulis, rhos_approx - self.training_data, optimize='greedy')
            elif self.error_type =='trace product':
                # for tr[sigma rho] tr[sigma rho']
                traces = -np.einsum('kij, nlji -> nlk', self.paulis, self.training_data, optimize = 'greedy')
            
            elif self.error_type =='wasserstein':
                # for tr[a_kl sigma (rho - rho')] (based on wasserstein grad descend)
                #_, weights[i,:] = wasserstein1(self.training_rho[i], rho_end, (self.paulis, self.pauli_id_list))
                #traces3 = weights[i,j]
                traces = np.tile(self.weights, (num_t+1, num_l, 1))
                
            else:
                print('WARNING: Error type {} not supported with pulse based gradient descend'.format(self.error_type))
                print('Switching to default "pauli trace"')
                self.error_type = 'pauli trace'
                traces = np.einsum('kij, nlji -> nlk', self.paulis, rhos_approx - self.training_data, optimize='greedy')
            
            
            # Amplify the weight on the last training data set (the steady state)
            traces[1,-1,:] = traces[1,-1,:]*self.steadystate_weight
            traces[2:,-1,:] = 0
            
# =============================================================================
#             # Alternatively, use all time evolutions of steady state
#             traces[:,-1,:] = traces[:,-1,:] *self.steadystate_weight
# =============================================================================
            
            # Calculate the product of trace * partial _ delta U matrix for all combinations
            paulis_Uext = np.kron(paulis_U, np.eye(2**(m+1)))
            rhos_approx_ext = np.kron(rhos_approx, self.state_00)
            for n1 in range(1,num_t+1): #repetitions on U
                for n2 in range(num_t+1-n1): #index for rho_n
                    matrices = np.einsum('kab, bc, lcd -> lkad', paulis_Uext[n1-1], U, rhos_approx_ext[n2], optimize = 'greedy')
                    #matrices = np.einsum('aij, jk, bkl -> abil', paulis_Uext[n1-1], U, rhos_approx_ext[n2], optimize = 'greedy')
                    eta_t_sum += np.einsum('lk, lkij -> ij', traces[n1+n2], matrices, optimize= 'greedy' )
            
            eta_t_sum = Udag @ eta_t_sum/ (num_l* num_t* num_k)
        
            # Set the actual gradient based on eta
            eta_t_sum_dag = np.conjugate(np.transpose(eta_t_sum))
            for t in range(self.Zdt):
                for k in range(0,num_controls):
                    updatepart = np.trace(-2j *self.control_H[k,1].full() @ (U_full[t].full() @ (eta_t_sum - eta_t_sum_dag) @U_full[t].dag().full() ) )
                    gradient[k,t,0] = self.lambdapar *theta[k,t,0] - np.real(updatepart)
                    gradient[k,t,1] = self.lambdapar *theta[k,t,1] - np.imag(updatepart)
            
            return gradient
        
        else:
            theta_p = theta.copy()
            theta_m = theta.copy()
            grad_theta = np.zeros(theta.shape)
            
            if self.n_grad_directions != -1:
                optimize_indices = rd.sample(list(range(len(theta))), self.n_grad_directions)
            else:
                optimize_indices = range(len(theta))
                
            for i in optimize_indices:
                theta_p[i] = theta_p[i] + eps
                theta_m[i] = theta_m[i] - eps
                if math.isnan(theta_p[i]) or math.isnan(theta_m[i]):
                    print("component {} gives a nan".format(i),theta_p[i], theta_m[i])
                grad_theta[i] = np.real(self.training_error(theta_p) - self.training_error(theta_m))/(2*eps)
                theta_p[i] = theta_p[i] - eps
                theta_m[i] = theta_m[i] + eps
                
            return grad_theta
    
    
    def _armijo_update(self, theta, sigmas, grad_theta, gamma = 10**(-4)):
        """
        Run a single armijo step for a given set of parameters and gradient

        Parameters
        ----------
        theta : np.array
            parameters of unitary circuit
        sigmas : tuple of floats
            (sigmabig, sigmasmall, sigmastart) to iteratively determine an
            optimal starting stepsize.
        grad_theta : np.array
            gradient in the parameters.
        gamma : float, optional
            armijo parameter. The default is 10**(-4).

        Returns
        -------
        update_theta : np.array
            Optimal updated theta.
        sigmas : tuple of floats
            Updated tuple of sigmas.
        grad_zero : bool
            Bool to signal vanishing gradient.

        """
        
        (sigmabig, sigmasmall, sigmastart) = sigmas
        
        if sigmabig >= 3: #Reduce initial step size if consistently to big
            sigmastart =sigmastart/2
            sigmabig = 0
        if sigmasmall >= 3: #Increase initial step size if consistently to small
            sigmastart = sigmastart*2
            sigmasmall = 0
        
        #Initialize inner loop parameters
        descended=False
        sigma = sigmastart
        fid = self.training_error(theta)
        first=True
        
        #Armijo stepsize rule update
        grad_zero = False
        while not descended:
                
            update_theta = theta -sigma*grad_theta
            
            update_fid = self.training_error(update_theta, weights = self.weights)
     
            if update_fid -fid < -(gamma*sigma*np.sum(np.multiply(grad_theta, grad_theta))):
                descended=True
                if first:
                    sigmasmall = sigmasmall + 1
            elif sigma<10**-10: #or update_fid - fid ==0:
                descended=True
                print("small sigma")
                grad_zero = True
                #count = max_count-1
            else:
                sigma=sigma/2
                if first:
                    sigmabig = sigmabig + 1
            first=False
        
        return update_theta, (sigmabig, sigmasmall, sigmastart), grad_zero
    
    def run_armijo(self, theta, max_count, gamma = 10**(-4), sigmastart=1,
                   epsilon = 0.01):
        """
        Function to run the full armijo gradient descend. 
        solution saved as self.theta_opt

        Parameters
        ----------
        theta : np.array 
            initial parameters
        max_count : int
            Max gradient steps.
        gamma : float, optional
            armijo step parameter. The default is 10**(-4).
        sigmastart : float, optional
            initial step size. The default is 1.
        epsilon : float, optional
            step size for finite difference for gate based. The default is 0.01.

        Returns
        -------
        None.

        """
        error = np.ones([max_count])
        grad_size = np.zeros(max_count)
        
        if self.circuit_type == 'pulse based':
            theta = np.reshape(theta, (self.control_H.shape[0], self.Zdt, 2))
        
        # Set armijo parameters
        sigmabig = 0
        sigmasmall = 0
        sigmastart = sigmastart
        sigmas = (sigmabig, sigmasmall, sigmastart)
        
        # Set timing parameters
        time_grad = 0
        time_armijo = 0
        time_start = time.time()
        
        # Run update steps
        count = 1
        grad_zero = False
        while count < max_count and not grad_zero and error[count-1] > 10**(-6):
            
            error[count] = self.training_error(theta, weights = 0, error_type = 'pauli trace')
            
            # Calculate the weights for the wasserstein optimization
            if self.error_type == 'wasserstein':
                error_temp = self.training_error(theta)
            
            time0 = time.time()
            grad_theta = self.find_gradient(theta, eps = epsilon)
            grad_size[count] = np.inner(np.ravel(grad_theta),np.ravel(grad_theta))
            time1 = time.time()
            time_grad += time1 - time0
            
            
            
            theta, sigmas, grad_zero = self._armijo_update(theta, sigmas, grad_theta, gamma)
            self.theta_opt = theta
            if count%5==0 or count == max_count-1:
                print("Iteration ", count)
                print("   Max gradient term: ", np.amax(grad_theta))
                print("   Current error: ", error[count])
                print("   Current sigma values: ", sigmas)
                
                theta1, _ = self.reshape_theta_phi(np.array(theta))
                if self.circuit_type == 'pulse based':
                    plt.figure()
                    for k in range(2*self.m+1):
                        colours = ['b', 'r', 'g', 'darkorchid', 'gold', 'k']
                        plt.plot(np.linspace(0,self.T_pulse, self.Zdt), theta1[k,:,0], '-', color = colours[k%6], label = 'qubit {}'.format(k))
                        plt.plot(np.linspace(0,self.T_pulse, self.Zdt), theta1[k,:,1], ':', color = colours[k%6])
                        if self.control_H.shape == 2*(2*self.m+1):
                            plt.plot(np.linspace(0,self.T_pulse, self.Zdt), theta1[2*self.m+1+k,:,0], '--', color = colours[k%6])
                            plt.plot(np.linspace(0,self.T_pulse, self.Zdt), theta1[2*self.m+1+k,:,1], '-.', color = colours[k%6])
                    plt.legend()
                    plt.title('Iteration {}'.format(count))
                    plt.show()
            time2 = time.time()
            time_armijo += time2 - time1
            count +=1
        print('-----')
        print("Grad calculation time: ", time_grad, ' Armijo calculation time: ', time_armijo)
        print("Total grad descend time: {}:{:.2f}".format(int((time2- time_start)//60), ((time2-time_start)%60)))
            
        if count < max_count:
            error[count:] = 0
            grad_size[count:] = 0
        
        self.theta_opt = theta
        self.error = error
        
    def evolution_n(self, n, rho):
        """
        Calculate the exact evolution for n time steps of rho

        Parameters
        ----------
        n : int
            number of time steps.
        rho : np.array, 2**m x 2**m
            density matrix.

        Returns
        -------
        rho_end : np.array, n+1 x 2**m x 2**m
            list of evolutions of rho.

        """
        rho_end = np.zeros((n+1,2**self.m,2**self.m), dtype = np.complex128)
        rho_end[0] = rho
        for i in range(n):
            rho = self.evolution(rho)
            rho_end[i+1] = rho
        return rho_end
    
    @time_wrapper
    def unitary_approx_n(self, n, rho, U=np.array([None])):
        """
        Get Phi_kt(rho) = Phi_t(Phi_t(...Phi_t(rho)...) for k = 0 till k = n

        Parameters
        ----------
        n : int
            Number of repetitions of the circuit.
        rho : np.array, 2**m x 2**m
            density matrix to start with.
        U : np.array, 2**(2*m+1) x 2**(2*m+1)
            Unitary evolution on the bigger system
            Gets internal if not specified

        Returns
        -------
        rho : np.array, n+1 x 2**m x 2**m
            density matrix after [1,...,n] evolutions

        """

        rho_end = np.zeros((n+1, 2**self.m, 2**self.m), dtype = np.complex128)    
        rho_end[0] = rho

        if (U == None).any():
            try:
                theta, gate_par = self.reshape_theta_phi(self.theta_opt)
                U = self.circuit.gate_circuit(theta = theta, n=self.repeat, gate_par = gate_par)
            except AttributeError:
                print("No optimal theta found and no unitary specified")
                return rho_end 
        
        Udag = np.transpose(np.conjugate(U))
        for i in range(n):
            UrhoU = U @ np.kron(rho,self.state_00) @ Udag
            rho = np.trace(UrhoU.reshape(2**self.m,2**(self.m+1),2**self.m,2**(self.m+1)),axis1 = 1, axis2=3)
            rho_end[i+1] = rho
            
        return rho_end
        
    def reshape_theta_phi(self, theta_phi):
        """
        Reshape a 1d array of theta parameters into the correct shape
        n_controls x Zdt x 2 for pulse based
        depth x 2*m+1 x 3 + entanglement gate parameters for gate based

        Parameters
        ----------
        theta_phi : np.array
            1 dimensional array of parameters.

        Returns
        -------
        theta : np.array
            array of rotation parameters.
        gate_par : np.array
            array of entanglemant gate parameters.

        """
        gate_par = 0
        if self.circuit_type == 'pulse based':
            theta = np.reshape(theta_phi, (self.n_controls, self.Zdt, 2))
        else:
            theta_mindex = self.depth*(2*self.m+1)*3
            theta = np.reshape(theta_phi[0:theta_mindex], (self.depth, 2*self.m+1, 3))
            if self.circuit_type == 'xy' or self.circuit_type == 'decay':
                n_pars = len(theta_phi[theta_mindex:])//(self.depth-1)
                gate_par = np.reshape(theta_phi[theta_mindex:],(self.depth-1, n_pars))
            elif self.circuit_type == 'ryd':
                gate_par = np.reshape(theta_phi[theta_mindex:],(self.depth-1, 1))
        return theta, gate_par
    
    def set_steady_state(self):
        """
        set self.steady_state as steady state of the system

        Returns
        -------
        None.

        """
        random_ket = qt.rand_ket_haar(dims = [[2**self.m], [1]])
        random_ket.dims = [[2]*self.m,[2]*self.m]
        random_bra = random_ket.dag()
        steady_state_old = (random_ket * random_bra).full()
        steady_state_new = self.evolution(steady_state_old)
        count = 0
        maxcount = 1000
        while np.amax(np.abs(steady_state_old - steady_state_new)) > 10**(-6) and count < maxcount:
            steady_state_old = steady_state_new
            steady_state_new = self.evolution(steady_state_old)
            count+=1
        if count == maxcount:
            print("Steady state not found")
        self.steady_state = steady_state_new
    
if __name__ == '__main__':
    test = stinespring_unitary_update(m=2)
    t = 1
    
    m=2
    depth = 6                           # Depth of simulation circuit
    n_training = 10                     # Number of initial rho's to check
    seed = 1                            # Seed for random initial rho's
    theta = np.ones([depth, 2*m, 3])    # Initial theta guess
    max_it_training = 5                # Number of Armijo steps in the gradient descend
    
    # rho0
    rho0 = np.zeros([2**m,2**m])
    rho0[0,0] = 1
    
    # Pauli spin matrices
    Id = np.array([[1,0],[0,1]])
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,1j],[-1j,0]])
    Z = np.array([[1,0],[0,-1]])

    # Lindbladian
    gam0 = 1
    gam1 = 2
    An = np.array([[[0,0,0,0],[0,0,0,0],[0,0,gam1**(1/2),0],[0,0,0,0]], #|11> to |10>
                  [[0,0,0,0],[0,0,0,gam0**(1/2)],[0,0,0,0],[0,0,0,0]], #|11> to |01>
                  [[0,0,gam0**(1/2),0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], #|10> to |00>
                  [[0,gam1**(1/2),0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]  #|01> to |00>
                  ])

    # Hamiltonian
    om0 = 1
    om1 = 2
    H = om0*np.kron(X,Id) + om1*np.kron(Id,X)
    
    
    # Class checks
    test.set_original_lindblad(H, An, t)
    test.evolution(rho0)
    print("Lindblad added")
    
    test.set_training_data(n_training,seed)
    print("training data made,\n  training rho shape: ",test.training_rho.shape,
          "\n  Phi_t(rho) shape: ", test.training_data.shape)
    
    test.set_unitary_circuit()
    print("Unitary circuit added")
    test.circuit.gate_circuit(np.ones([depth, 2*m, 3]))
    
    test.training_error(theta)
    test.find_gradient(theta,0.01)
    
    test.run_all_lindblad(H, An, t, rho0, n_training, seed, depth, theta, max_it_training)
    
    
