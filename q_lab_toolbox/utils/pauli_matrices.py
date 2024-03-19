"""
Numpy arrays of the Pauli matrices Id, X, Y, Z.
In addition the zero matrix is defined.
SPIN_MATRIX_DICT has a dictionary that maps the strings
"Id", "X", "Y", "Z", "O" to their respective matrices.

Id = [[1, 0],
      [0, 1]]

X = [[0, 1],
     [1, 0]]

Y = [[0, -i],
     [i,  1]]

Z = [[1,  0],
     [0, -1]]

     
O = [[0, 0],
     [0, 0]]

References:
    Introduction to Quantum Mechanics, Griffiths and Schoeter
    (page 215 (?))

Info:
    Created on Thu Feb 22 2024

    @author: davidvrchen
"""
import numpy as np

Id = np.array(
    [[1, 0],
     [0, 1]]
    )

X = np.array(
    [[0, 1],
     [1, 0]]
    )

Y = np.array(
    [[0 , -1j],
     [1j,  0 ]]
     )

Z = np.array(
    [[1,  0],
     [0, -1]]
     )

O = np.array(
    [[0, 0],
     [0, 0]]
    )

spin_matrix_dict = {"I": Id, 'X': X, 'Y': Y, 'Z': Z, 'O': O}
