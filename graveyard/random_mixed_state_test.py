# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:20:27 2023

@author: lviss
"""

import numpy as np
import qutip as qt

m = 4

mix_factor = np.random.rand()**1/2

#mixed matrix
evals = np.random.normal(size = 2**m)
evals = evals**2/np.sum(evals**2)

#zero matrix
zero_mat = np.zeros((2**m,2**m))
zero_mat[0,0] = 1

# mixed matrix
init_matrix = mix_factor*zero_mat + (1-mix_factor)*np.diag(evals)
random_mixed = qt.Qobj(init_matrix, dims = [[2]*m, [2]*m])

U = qt.random_objects.rand_unitary_haar(N = 2**m, dims = [[2]*m, [2]*m])
random_mixed = U*random_mixed*U.dag()
