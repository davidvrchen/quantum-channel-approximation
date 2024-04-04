"""
Some common variables relating to Pauli spin matrices.
Contains:
    qt.Qobj objects for the Pauli matrices Id, X, Y, Z,
    in addition the zero matrix is defined.

    SPIN_MATRIX_DICT has a dictionary that maps the strings
    "Id", "X", "Y", "Z", "O" to their respective operators.

    SPIN_MATRICES_LST is a list ["I", "X", "Y", "Z"]

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

Info:
    Created on Thu Feb 22 2024

    Last update on Thu Apr 4 2024

    @author: davidvrchen
"""

import qutip as qt

Id = qt.Qobj(([[1, 0], [0, 1]]))

X = qt.sigmax()

Y = qt.sigmay()

Z = qt.sigmaz()

O = qt.Qobj([[0, 0], [0, 0]])

SPIN_MATRIX_DICT = {"I": Id, "X": X, "Y": Y, "Z": Z, "O": O}
SPIN_MATRICES_LST = ["I", "X", "Y", "Z"]
