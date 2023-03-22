# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:39:19 2022

@author: lviss
"""

import qutip as qt
import numpy as np
import scipy as sc
import multiprocessing as mp
import math
from itertools import product
from more_itertools import distinct_permutations
import random as rd
import re


def wasserstein1(rho1, rho2, pauli_tuple):
    """
    Maximizes SUM_j c_j w_j over w_j with 
    SUM_{j s.t. P_j acts as X,Y,Z on qubit i} |w_j|<=1   : for i =1,...,n
    and c_j = Tr[(rho1-rho2) P_j]
    with P_j pauli spin matrices given in pauli_tuple

    Parameters
    ----------
    rho1 : Qobj, matrix
        Density matrix 1.
    rho2 : Qobj, matrix
        Density matrix 2.
    pauli_tuple : (np.array, np.array)
        tuple with pauli spin matrices as np.arrays, and list of qubits it acts on.

    Returns
    -------
    max_expectation : float
        maximal expectation that was found.
    weights : np.array
        weights that maximize the expectation

    """
    paulis, id_qubit_list = pauli_tuple
    max_expectation = 0
    
    num_pauli = len(paulis)
    num_bit = id_qubit_list.shape[0]
    traces = np.zeros([num_pauli])
    for i in range(num_pauli):
        traces[i] = np.real((paulis[i]*(rho1-rho2)).trace())
    
    # Include the absolute value by setting -w'_j <= w_j <= w'_j
    # This gives |w_j| = w'_j. w'_j in [0,1], w_j in [-1,1]
    # parameter list is first all w'_j, then all w_j 
    # So #pars = 2 * #pauli-matrices 
    # And #constraints = m + #pars
    
    # Minimize MINUS w_j * P_j (rho1 - rho2) (is max over plus)
    obj = np.append(np.zeros(num_pauli),-traces)
    lhs_ineq = np.zeros([num_bit+2*num_pauli, 2*num_pauli])
    lhs_ineq[0:num_bit,0:num_pauli] = id_qubit_list
    for i in range(num_pauli):
        
        # -w'_i -w_i <= 0
        lhs_ineq[num_bit + 2*i,     i] = -1 
        lhs_ineq[num_bit + 2*i,     num_pauli +i] = -1
        
        # -w'_i +w_i <=0
        lhs_ineq[num_bit + 2*i +1,  i] = -1
        lhs_ineq[num_bit + 2*i +1,  num_pauli +i] = 1
        
    rhs_ineq = np.append(np.ones(num_bit)/2, np.zeros(2*num_pauli))
    bnd = [(0,0.5) for _ in range(num_pauli)]
    bnd = bnd + [(-0.5,0.5) for _ in range(num_pauli)]
    
    opt = sc.optimize.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd)
    
    if opt.success:
        max_expectation = -opt.fun
        x = opt.x
        weights = x[num_pauli:] #get non-absolute values.
        #p_print(opt.x)
        #p_print(max_expectation)
    else:
        p_print("Wasserstein optimum finding did not converge")
        max_expectation = 0
        weights = 0
        
    return max_expectation, weights


def unitary(h, t, lam, beta):
    m = len(h.dims[0])
    iden = qt.Qobj(np.eye(2**m),dims = [[2]*m,[2]*m])
    u =  (-1j * beta * (lam*iden + h)*t).expm()
    return u
    #return lambda rho0: (u*rho).tr()

def unitary_trot(h, psi_0, lam, dt, steps_per_dt, n_dt, analytic = False):
    m = len(h.dims[0])
    paulis = get_paulis(m, 'full')
    h_expansion = ham_expand_pauli(h, paulis[0])
    h_expansion[0] += lam
    #iden = qt.Qobj(np.eye(2**m),dims = [[2]*m,[2]*m])
    iden = np.eye(2**m)
    T = dt*n_dt
    
    # Calculate a single element of the trotterization 
    # by multiplying exp(pk) for all pauli spin matrices
    single_trot = iden.copy()
    for i in range(len(h_expansion)):
        exp_pk = np.cos(dt*h_expansion[i]/steps_per_dt)*iden \
                                   - 1j*np.sin(dt*h_expansion[i]/steps_per_dt)*paulis[0][i]
        single_trot = single_trot@exp_pk
        #print('{:.5f}'.format(np.sin(dt*h_expansion[i]/steps_per_dt)))
    
    # Calculating the unitary matrix for each iteration 
    U_single = np.linalg.matrix_power(single_trot,steps_per_dt)
    U_single = qt.Qobj(U_single, dims = [[2]*m, [2]*m])
    
    if analytic:
        U_single_exact = (1j*dt*(h- lam*qt.Qobj(iden, dims =[[2]*m,[2]*m]))).expm()
        U_single = U_single_exact
        print(np.max(np.abs(ham_expand_pauli(U_single - U_single_exact, paulis[0]))))

    # Calculate successive results of the trotterization
    psi_0_bra = psi_0.dag()
    psi_t = psi_0.copy()
    result = np.zeros(n_dt, dtype=np.csingle)
    for i in range(n_dt):
        result[i] = (psi_t.dag()*psi_0).tr()
        psi_t = U_single * psi_t
    return result


def exp_approx(coeff, paulis, cutoff_list, beta_list):
    print("Update max and min given for beta: {}".format(beta_list[0]))
    h_exp_approx = np.zeros([len(cutoff_list),len(beta_list),len(coeff)], dtype = np.csingle)
    mult_factor_list = np.ones(len(beta_list))
    
    # Initialize expansion of the identity matrix
    expansion_coeff = np.zeros(len(coeff),dtype = np.csingle)
    expansion_coeff[0] = 1
    c_index = 0
    for i in range(max(cutoff_list)):
        # Update approximation with next term in taylor series
        update = np.outer(mult_factor_list, expansion_coeff)
        h_exp_approx[-1,:,:] = h_exp_approx[-1,:,:] + update
        
        # If at a cutoff length, add to result
        if i in cutoff_list:
            h_exp_approx[c_index,:,:] = h_exp_approx[-1,:,:]
            c_index +=1
        
        #Update multiplication factor and pauli expansion
        mult_factor_list = -1* mult_factor_list * beta_list / (i+1)
        expansion_coeff = mult_paulis(expansion_coeff, coeff, paulis)
        
        # Print 
        print("exp() approximated up to order {} of {}".format(i, max(cutoff_list)-1), end = '\n')
        print("Max component update: {:.2e} and min: {:.2e}".format(max(np.real(update[0,:])), min(np.real(update[0,:]))))
        nonzero = sum(1 for gamma in h_exp_approx[-1,0,:] if abs(gamma) < 10**(-6))
        print("Nonzero components: {}".format(nonzero))
    return h_exp_approx

def power_paulis(coeff, paulis, power):
    binary_powers = format(power,'b')
    coeff_f = np.zeros(len(coeff), dtype = np.csingle)
    coeff_p = coeff.copy()
    first = True
    if power ==0:
        coeff_f[0] = 1
        return coeff_f
    elif power <0:
        print('negative powers not supported in power_paulis')
        raise ValueError
        
    for p in binary_powers[::-1]:
        if p == '1':
            if first:
                coeff_f[:] = coeff_p[:]
                first = False
            else:
                coeff_f = mult_paulis(coeff_p, coeff_f, paulis)
        coeff_p = mult_paulis(coeff_p, coeff_p, paulis)
        #print('binary power ',p, ': ', type(coeff_f[0]))
    return coeff_f
        
mult_dict = {'II': (1,'I'), 'XX': (1,'I'), 'YY':(1,'I'), 'ZZ':(1,'I'),
             'IX': (1,'X'), 'XI': (1,'X'), 'IY': (1,'Y'), 'YI': (1,'Y'), 
             'IZ': (1,'Z'), 'ZI': (1,'Z'), 'XY': (1j,'Z'), 'YX': (-1j,'Z'),
             'YZ': (1j,'X'), 'ZY': (-1j,'X'), 'ZX': (1j,'Y'), 'XZ': (-1j,'Y')}

def mult_paulis(coeff1, coeff2, paulis):
    coeff_f = np.zeros(np.shape(coeff1), dtype = np.csingle)
    for i, p1 in enumerate(paulis):
        for j, p2 in enumerate(paulis):
            pauli_sign = [mult_dict[p1[k]+p2[k]] for k in range(len(p1))]
            sign = np.prod([pauli[0] for pauli in pauli_sign])
            pauli_new = [pauli[1] for pauli in pauli_sign]
            index = np.where((paulis == pauli_new).all(axis=1))[0]
            coeff_f[index] = coeff_f[index] + coeff1[i]*coeff2[j]*sign
# =============================================================================
#             if pauli_new == ['X', 'X']:
#                 print("Pauli 1 {}, Pauli 2 {}, Pauli new {}".format(p1, p2, pauli_new))
#                 print("coeff1 {:.5f}, coeff 2 {:.5f}, coeff new {:.5f}".format(coeff1[i], coeff2[j], coeff1[i]*coeff2[j]*sign))
# =============================================================================
# =============================================================================
#             if coeff1[i]!=0 and coeff2[j]!=0:
#                 print("Pauli 1 {}, Pauli 2 {}, Pauli new {}".format(p1, p2, pauli))
#                 print("coeff1 {}, coeff 2 {}, coeff new {}".format(coeff1[i], coeff2[j], coeff1[i]*coeff2[j]*sign))
# =============================================================================
    return coeff_f

def partial_trace(mat, m, n_trace):
    """
    Calculate the partial trace of a system with m qubits, tracing out the qubits
    specified in n_trace

    Parameters
    ----------
    mat : np.array, dims = 2**m, 2**m
        matrix describing the quantum system that one wants to trace out the 
        qubits over.
    m : int
        number of qubits.
    n_trace : list of int
        list of indices of qubits to trace out.

    Returns
    -------
    mat : np.array, dims = 2**(m-len(n_trace)), 2**(m-len(n_trace))
        resulting matrix with traced out qubits.

    """
    n_trace = np.sort(n_trace)
    m_new = m - len(n_trace)
    if mat.shape[0] != 2**m:
        print("Matrix dimensions do not match")
        return mat
    mat = np.reshape(mat, tuple([2]*(2*m)))
    for i, n in enumerate(n_trace):
        mat = np.trace(mat, axis1 = n - i, axis2 = m + n - 2*i)
        
    mat = np.reshape(mat, (2**(m_new), 2**(m_new)))
    return mat

def ham_expand_pauli(h_mol, pauli_mat):
    """
    

    Parameters
    ----------
    h_mol : qt.Qobj, dims = [[2]*m, [2]*m]
        Hamiltonaion as qutip quantum object.
    paulis : np.array, dims = (K, 2**m, 2**m)
        numpy array of all pauli matrices.

    Returns
    -------
    h_expansion : np.array, dims = K
        numpy array of coefficients a_k s.t. h_mol = sum_k^K a_k paulis_k.
        
    To Do
    - Faster decomposition algorithm.

    """
    h = h_mol.full()
    m = int(round(math.log2(pauli_mat.shape[1])))
    product = h @ pauli_mat
    h_expansion = np.trace(product, axis1=1, axis2=2)
    return h_expansion* (2**(-m))

def truncate_pauli(pauli_name, n_trunc):
    """
    Sets the pauli spin matrices to I for all qubits starting at n_trunc

    Parameters
    ----------
    pauli_name : np.ndarray, dims = [num_paulis, m]
        list of str in {I, X, Y, Z} for the individual pauli spin matrices per qubit
    n_trunc : int
        index of qubit to truncate on, start count at 0.

    Returns
    -------
    pauli_names_trunc: np.array

    """
    k = pauli_name.shape[0]
    m = pauli_name.shape[1]
    pauli_names_trunc = pauli_name
    if n_trunc < m:
        pauli_names_trunc[:,n_trunc:] = 'I'
    
    pauli_list = np.ndarray([k, 2**m, 2**m], dtype = np.complex_)
    for i, name in enumerate(pauli_names_trunc):
        pauli_list[i,:,:] = pauli_from_str(name)
    return pauli_list



def get_paulis(m, space = 'full'):
    """
    Generates a set of pauli matrices to be optimized over in the calculation 
    of the wasserstein 1 distance

    Parameters
    ----------
    m : int
        number of qubits.
    space : str, optional
        name of specific subset of spin matrices to use.
        options: 'full', 'order {}'.format(k)
        The default is 'full'. For unrecognised types, 'full' is used as well.

    Returns
    -------
    pauli_list : np.ndarray, dims = [num_paulis, 2**m, 2**m]
        array of all selected pauli spin density matrices on m qubits.
    pauli_list_names : np.ndarray, dims = [num_paulis, m]
        list of str in {I, X, Y, Z} for the individual pauli spin matrices per qubit

    """
    
    pauli_single = ['I', 'X', 'Y', 'Z']
    
    if 'order' in space:
        order = int(space[-1])
        num_pauli = sum(3**i * math.comb(m,i) for i in range(order+1))
        i=0
        pauli_list_names = []
        for k in range(order+1):
            terms_pauli = ['XYZ' for i in range(k)] + ['I' for i in range(m-k)]
            terms_permuted = list(distinct_permutations(terms_pauli))
            for name_list in terms_permuted:
                pauli_list_names = pauli_list_names + list(product(*name_list))
    elif 'random' in space:
        num_pauli = int(re.findall('\d+\Z', space)[0])
        pauli_list_names_full = list(product(pauli_single, repeat = m))
        pauli_list_names = rd.sample(pauli_list_names_full, num_pauli)
    else:
        if space != 'full':
           p_print('Pauli subspace "{}" not recognised, using ALL pauli matrices'.format(space))
        num_pauli = 4**m
        pauli_list_names = list(product(pauli_single, repeat = m))
    
    pauli_list_names = np.array(pauli_list_names)
    
    pauli_list = np.ndarray([num_pauli, 2**m, 2**m], dtype = np.complex_)
    id_qubit_list = np.zeros([m, num_pauli])
    for i, name in enumerate(pauli_list_names):
        pauli_list[i,:,:] = pauli_from_str(name)
        for j in range(m):
            if name[j] != "I":
                id_qubit_list[j,i]=1
                
    return pauli_list, pauli_list_names, id_qubit_list


def pauli_from_str(pauli_names):
    
    identity = np.array([[1,0],[0,1]])
    sigmax = np.array([[0,1],[1,0]])
    sigmay = np.array([[0, -1j],[1j,0]])
    sigmaz = np.array([[1,0],[0,-1]])
    zero = np.array([[0,0],[0,0]])
    
    spin_m_dict = {"I": identity, 'X': sigmax, 'Y': sigmay, 'Z': sigmaz, 'O': zero}
    
    matrix = np.ones([1,1])
    for name in pauli_names:
        if name in ['I', 'X', 'Y', 'Z', 'O']:
            matrix = np.kron(matrix, spin_m_dict[name])
        else:
           p_print("Pauli matrix type not found")
    return matrix

def p_print(text, *args, **kwargs):
   print('{}: {}'.format(mp.current_process().name, text), *args, **kwargs)

def generate_hamiltonian(m):
    """
    Generates a random problem Hamiltonian

    Parameters
    ----------
    m : int
        number of qubtis.

    Returns
    -------
    Hmol : Qobj, 2**m x 2** m 
        Problem Hamiltonian of which the ground state energy is to be found
    Hmol_groundenergy : float
        Lowest eigenenergy of Hmol

    """
    b = np.random.random_integers(-2000, 2000, size=(2 ** m, 2 ** m)) / 2000
    b_symm = (b + b.T) / 2
    Hmol = qt.Qobj(b_symm)
    Hmol.dims = [[2] * m, [2] * m]
    Hmol_energies = Hmol.eigenenergies()
    p_print("Lowest eigenenergies of the hamiltonian: \n{}".format(Hmol_energies[0:2]))
    Hmol_groundenergy = Hmol_energies[0]
    return Hmol, Hmol_groundenergy

def gradientnewnn(Z, Hmol, rho0, U, lambdapar, control_H, qubit = []):
    """
    Calculates gradient of energy w.r.t. pulse parameters

    Parameters
    ----------
    Z : np.ndarray complex, num_control x Zdt x 2
        Describes the (complex) pulse parameters of the system
    Hmol : Qobj, 2**m x 2** m 
        Problem Hamiltonian of which the ground state energy is to be found
    rho0 : Qobj, 2**m x 2**m
        Density matrix describing initial state
    U : np.ndarray Qobjs, Zdt
        Array describing unitary evolution at each timestep
    lambdapar : float
        regularization parameter for J2 term
    control_H : np.ndarray Qobj 2**m x 2**m, num_control x 2 
        array of Qobj describing the control operators
    qubit : list
        list of indices of control_H to find the gradient for, 
        corresponds with qubit index
        If [], find gradient over all control_H.

    Returns
    -------
    gradient : np.ndarray, num_control x Zdt x2
        Describes the complex gradient of the energy w.r.t. different control parameters

    """
    num_controls=len(control_H[:,0])
    gradient=np.zeros([num_controls,len(Z[0,:,0]),2])
    
    # Set the qubits to take the gradient over
    full_range = list(range(0, num_controls))
    if len(qubit)==0:
        qubit = full_range
    else:
        qubit = [value for value in qubit if value in full_range]
        
    # Calculate the gradient
    for t in range(0,len(Z[0,:,0])):
        for k in qubit:
                updatepart=control_H[k,1]*(-2j*U[t]*U[-1].dag()*rho0*U[-1]*Hmol*U[t].dag()+2j*U[t]*Hmol*U[-1].dag()*rho0*U[-1]*U[t].dag())
                update=2*lambdapar*(Z[k,t,0]+1j*Z[k,t,1])-updatepart.tr()
                gradient[k,t,0]=np.real(update)
                gradient[k,t,1]=np.imag(update)
    return gradient

def Znorm(Z,T):
    """
    Determines the L2 norm of the Zs

    Parameters
    ----------
    Z : np.ndarray complex, num_control x Zdt x 2
        Describes the (complex) pulse parameters of the system
    T : float
        Total evolution time.

    Returns
    -------
    float
        L2 norm of Z.

    """
    norm=0
    for t in range(len(Z[0,:,0])):
        norm=norm+np.linalg.norm(Z[:,t,:], 'fro')**2
    return math.sqrt(1/len(Z[0,:,0])*norm*T)

def create_driving_hamiltonians(m, interaction, structure):
    """
    Creates the driving Hamiltonian describing the drift of the system

    Parameters
    ----------
    m : int
        number of qubits.
    type : string
        selection of drive Hamiltonian.

    Returns
    -------
    Hamiltonian : QObj, 2**m x 2**m
          Hamiltonian describing the natural drift of the system

    """
    
    Hamiltonian = qt.Qobj(dims=[[2] * m, [2] * m])
    project1111 = qt.Qobj(np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]), dims = [[2]*2,[2]*2])
    project0110_1001 = qt.Qobj(np.array([[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,0]]), dims = [[2,2],[2,2]])
    
    pairs_distance = generate_gate_connections(m, structure = structure, cutoff = False)
        
    if interaction=='basic11':
        for i, j, d in pairs_distance:
            if d <=1:
                Hamiltonian=Hamiltonian +qt.qip.operations.gates.expand_operator(project1111,m,[i,j])
        return Hamiltonian
    
    elif interaction=='rydberg11':
        for i, j, d in pairs_distance:
            Hamiltonian += d**(-3) *qt.qip.operations.gates.expand_operator(project1111,m,[i,j])         
        return Hamiltonian
    
    elif interaction == 'dipole0110':
        for i, j, d in pairs_distance:
            Hamiltonian += d**(-3/2) *qt.qip.operations.gates.expand_operator(project0110_1001,m,[i,j])
        return Hamiltonian
    
    else:
        raise ValueError(interaction +' is not a specified driving Hamiltonian interaction')

# =============================================================================
#     elif type=='dipole0110':
#         Hamiltonian = qt.Qobj(dims=[[2] * m, [2] * m])
#         for i in range(m):
#             for j in range(i+1,m):
#                 Hamiltonian=Hamiltonian+1/((j-i)**3)*(project0110op(i,j,m)+project1001op(i,j,m))
#         return Hamiltonian
#     elif type=='pairwise11':
#         Hamiltonian = qt.Qobj(dims=[[2] * m, [2] * m])
#         for i in range(m - 1):
#             Hamiltonian = Hamiltonian + project1111op(i, i + 1, m)
#         return Hamiltonian
#     elif type=='pertwo11':
#         Hamiltonian = qt.Qobj(dims=[[2] * m, [2] * m])
#         for i in range(np.int(np.round(m /2))):
#             Hamiltonian = Hamiltonian + project1111op(2*i, 2*i + 1, m)
#         return Hamiltonian
#     elif type=='basicrr':
#         Hamiltonian = qt.Qobj(dims=[[3] * m, [3] * m])
#         for i in range(m):
#             for j in range(i+1,m):
#                 Hamiltonian=Hamiltonian+projectrrrrop(i,j,m)
#         return Hamiltonian
# =============================================================================
    

def create_control_hamiltonians(m,type_h):
    """
    Creates the control Hamiltonian operators

    Parameters
    ----------
    m : int
        number of qubits.
    type_h : string
        describes the type of control Hamiltonian.

    Returns
    -------
    Hamiltonians : np.ndarray Qobj's, num_controls x 2
        array of control Hamiltonians Ql to be influenced

    """
    if type_h=='rotations':
        Hamiltonians=np.ndarray([m,2,],dtype=object)
        project10op = qt.Qobj(np.array([[0,0],[1,0]]),dims = [[2],[2]])
        project01op = qt.Qobj(np.array([[0,1],[0,0]]),dims = [[2],[2]])
        for k in range(m):
            Hamiltonians[k,0]=qt.qip.operations.gates.expand_operator(project10op, m, k)
            Hamiltonians[k,1]=qt.qip.operations.gates.expand_operator(project01op, m, k)
        return Hamiltonians
    elif type_h=='realrotations':
        Hamiltonians=np.ndarray([m,2,],dtype=object)
        project10op = qt.Qobj(np.array([[0,0],[1,0]]),dims = [[2],[2]])
        project01op = qt.Qobj(np.array([[0,1],[0,0]]),dims = [2,2])
        for k in range(m):
            gate = qt.qip.operations.gates.expand_operator(project01op, m, k)
            gate += qt.qip.operations.gates.expand_operator(project10op, m, k)
            Hamiltonians[k,0]=gate
            Hamiltonians[k,1]=gate
        return Hamiltonians
    
    elif type_h=='rotations+11':
        Hamiltonians=np.ndarray([2*m,2,],dtype=object)
        project10op = qt.Qobj(np.array([[0,0],[1,0]]),dims = [[2],[2]])
        project01op = qt.Qobj(np.array([[0,1],[0,0]]),dims = [[2],[2]])
        project11op = qt.Qobj(np.array([[0,0],[0,1]]), dims = [[2],[2]])
        for k in range(m):
            Hamiltonians[k,0]=qt.qip.operations.gates.expand_operator(project10op, m, k)
            Hamiltonians[k,1]=qt.qip.operations.gates.expand_operator(project01op, m, k)
            Hamiltonians[m+k,0]=qt.qip.operations.gates.expand_operator(project11op, m, k)
            Hamiltonians[m+k,1]=qt.qip.operations.gates.expand_operator(project11op, m, k)
        return Hamiltonians
    
# =============================================================================
#     elif type_h=='rotations+singledipole':
#         Hamiltonians=np.ndarray([m+1,2,],dtype=object)
#         for k in range(m):
#             Hamiltonians[k,0]=project10op(k,m)
#             Hamiltonians[k,1]=project01op(k,m)
#         Hamiltonians[m, 0]=qt.Qobj(dims=[[2]*m,[2]*m])
#         Hamiltonians[m, 1]=qt.Qobj(dims=[[2]*m,[2]*m])
#         for k in range(m):
#             for l in range(k+1,m):
#                 Hamiltonians[m, 0]=Hamiltonians[m, 0]+project1001op(k,l,m)
#                 Hamiltonians[m, 1]=Hamiltonians[m, 1]+project0110op(k,l,m)
#         return Hamiltonians
#     elif type_h=='rotations+XX':
#         Hamiltonians=np.ndarray([m+1,2,],dtype=object)
#         for k in range(m):
#             Hamiltonians[k,0]=project10op(k,m)
#             Hamiltonians[k,1]=project01op(k,m)
#         Hamiltonians[m, 0]=qt.Qobj(dims=[[2]*m,[2]*m])
#         Hamiltonians[m, 1]=qt.Qobj(dims=[[2]*m,[2]*m])
#         for k in range(m):
#             for l in range(k+1,m):
#                 Hamiltonians[m, 0]=Hamiltonians[m, 0]+XXop(k,l,m)
#                 Hamiltonians[m, 1]=Hamiltonians[m, 1]+XXop(k,l,m)
#         return Hamiltonians
#     elif type_h=='realrotations+11':
#         Hamiltonians=np.ndarray([2*m,2,],dtype=object)
#         for k in range(m):
#             Hamiltonians[k,0]=project10op(k,m)+project01op(k,m)
#             Hamiltonians[k,1]=project01op(k,m)+project10op(k,m)
#             Hamiltonians[m+k,0]=project11op(k,m)
#             Hamiltonians[m+k,1]=project11op(k,m)
#         return Hamiltonians
#     elif type_h=='rotations+singledipole+11':
#         Hamiltonians=np.ndarray([2*m+1,2,],dtype=object)
#         for k in range(m):
#             Hamiltonians[k,0]=project10op(k,m)
#             Hamiltonians[k,1]=project01op(k,m)
#             Hamiltonians[m+k,0]=project11op(k,m)
#             Hamiltonians[m+k,1]=project11op(k,m)
#         Hamiltonians[2*m, 0] = qt.Qobj(dims=[[2] * m, [2] * m])
#         Hamiltonians[2*m, 1] = qt.Qobj(dims=[[2] * m, [2] * m])
#         for k in range(m):
#             for l in range(k + 1, m):
#                 Hamiltonians[2*m, 0] = Hamiltonians[2*m, 0] + project1001op(k, l, m)
#                 Hamiltonians[2*m, 1] = Hamiltonians[2*m, 1] + project0110op(k, l, m)
#         return Hamiltonians
#     elif type_h=='qutritrotations':
#         Hamiltonians=np.ndarray([m*2,2,],dtype=object)
#         for k in range(m):
#             Hamiltonians[k,0]=project10tritop(k,m)
#             Hamiltonians[k,1]=project01tritop(k,m)
#             Hamiltonians[m+k,0]=projectr1tritop(k,m)
#             Hamiltonians[m+k,1]=project1rtritop(k,m)
#         return Hamiltonians
# =============================================================================
    else:
        raise ValueError(type_h+' is not a specified way of creating control Hamiltonians.')
        
def generate_gate_connections(m, structure = 'triangle d=1', cutoff = True):

    pairs = []
    if 'd=' in structure:
        dist_scale = float(structure[structure.find('d=')+2:])
    elif 'd =' in structure:
        dist_scale = float(structure[structure.find('d =')+3:])
    else:
        dist_scale = 1
    
    if 'triangle' in structure:
        for i in range(m-1):
            for j in range(i+1,m):
                if i < m//2 and j >= m//2:
                    d_hor = abs(j -i -m//2 -1/2)
                    d_ver = 3**(1/2)/2
                else:
                    d_hor = j-i
                    d_ver = 0
                dist = dist_scale**2 *(d_hor**2 +d_ver**2)
                if dist <= 1.01:
                    pairs.append((i, j, dist))
                elif not cutoff:
                    pairs.append((i,j,dist))
                
    elif 'loose_pairs' in structure:
        for i in range(m-1):
            for j in range(i+1,m-1):
                if i < m//2 and j >= m//2:
                    d_hor = 2*abs(j-i-m//2)
                    d_ver = 1
                else:
                    d_hor = 2*(j-i)
                    d_ver = 0
                dist = dist_scale**2 *(d_hor**2 +d_ver**2)
                if dist <= 1.01:
                    pairs.append((i, j, dist))
                elif not cutoff:
                    pairs.append((i,j,dist))
    
    elif 'pairs' in structure:
        for i in range(m-1):
            for j in range(i+1,m-1):
                if i < m//2 and j >= m//2:
                    d_hor = abs(j-i-m//2)
                    d_ver = 1
                else:
                    d_hor = j-i
                    d_ver = 0
                dist = dist_scale**2 *(d_hor**2 +d_ver**2)
                if dist <= 1.01:
                    pairs.append((i, j, dist))
                elif not cutoff:
                    pairs.append((i,j,dist))
            if i >= m//2:
                pairs.append((i, m-1, 1))
    
    elif 'line' in structure:
        for i in range(m-1):
            for j in range(i+1,m):
                if i < m//2 and j >= m//2:
                    d_hor = 2*abs(j -i -m//2 -1/2)
                else:
                    d_hor = 2*(j-i)
                dist = dist_scale**2 *d_hor**2
                if dist <= 1.01:
                    pairs.append((i, j, dist))
                elif not cutoff:
                    pairs.append((i,j,dist))
    else:
        raise ValueError(structure + ' is not a specified atomic structure')
    
    return pairs