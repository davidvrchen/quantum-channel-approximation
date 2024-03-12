"""
Provides a function to construct the Lindbladian differential equation.

References:
    original code for decay_examply by @lviss

Info:
    Created on Mon March 11 2024

    @author: davidvrchen
"""

import numpy as np


def lindbladian(H, An):
    """
    Create the Lindbladian differential equation.

    Parameters:
    -----------

    H: Hamiltonian

    An: Jump operators
    """

    shape = H.shape

    print(shape)

    def _lindbladian(t, rho):
        rho = np.reshape(rho, (2**m, 2**m))
        result = -1j * (H @ rho - rho @ H)
        for A in An:
            Ad = np.conj(np.transpose(A))
            result = result + A @ rho @ Ad - Ad @ A @ rho / 2 - rho @ Ad @ A / 2
        result = np.reshape(result, 4**m)
        return result

    return _lindbladian

