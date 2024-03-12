"""
Functions needed to solve Lindbladian based on ReferenceSolverSettings.

References:
    Original code by @lviss

Info:
    Created on Tue Mar 12 2024

    @author: davidvrchen
"""

from dataclasses import dataclass

import numpy as np
import scipy as sc


@dataclass
class ReferenceSolverSettings:
    """Settings that define how to solve Lindbladian
    using the reference_solver algorithm.
    """

    t_steps_size: float

    def __post_init__(self):
        """Check if the settings are valid"""
        pass



def lindbladian(H, An):
    """Creates the RHS of the Lindbladian defined as
    i d_t rho = [H,rho] + sum_k An[k] rho An[k]* - 1/2 [An[k]*An[k], rho]
    { comment from original code }
    { but we want to work towards d_t rho = f(rho) }
    { so more like ... }
    d_t rho = -i([H,rho] + sum_k {An[k] rho An[k]* - 1/2 [An[k]*An[k], rho]) )



    Parameters:
    -----------
    H: Hamiltonian

    An: jump operators


    Note: get rid of the reshaping step if possible
    so the calc. has no explicit dep. on m
    """

    def _lindbladian(rho):
        """Given rho, computes the RHS of the Lindbladian
        the RHS of the """

        rho = np.reshape(rho,(2**m,2**m)) # not sure what the reshaping is for

        result = -1j*(H@rho - rho@H)

        for A in An:
            Ad = np.conj(np.transpose(A))
            result = result + A @rho @Ad - Ad @A @rho/2 - rho @Ad @A/2

        result = np.reshape(result,4**m) # not sure what the reshaping is for

        return result
    
    return _lindbladian




def solve_lindbladian(lindbladian, # lindbladian
                      s: ReferenceSolverSettings # settings
                      ): # -> should return ts, rhos
    """Solve lindbladian based on settings"""

    def inner():
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
    



# original code related to lindbladian
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