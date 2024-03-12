"""
Provides a function to construct the jump operators
from TargetSystemSettings


References:
    original code for decay_examply by @lviss

Info:
    Created on Mon March 11 2024

    @author: davidvrchen
"""

import numpy as np

from .settings import TargetSystemSettings


def jump_operators(s: TargetSystemSettings) -> np.ndarray:
    """Construct jump operators needed for the Lindbladian from settings."""

    m = s.m  # for easy of notation save the number of qubits

    if m == 1:
        (gam0,) = s.gammas  # comma needed to unpack the tuple
        y0 = gam0 ** (1 / 2)

        # |1> to |0>, |0> to |0
        A0 = [
            [0, y0],
            [0, 0],
        ]

        return np.array([A0])

    if m == 2:
        gam0, gam1 = s.gammas

        y0 = gam0 ** (1 / 2)
        y1 = gam1 ** (1 / 2)

        # |. 1> to |. 0>
        A0 = [
            [0, y0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, y0],
            [0, 0, 0, 0],
        ]

        # |1 .> to |0 .>
        A1 = [
            [0, 0, y1, 0],
            [0, 0, 0, y1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        return np.array([A0, A1])

    if m == 3:
        gam0, gam1, gam2 = s.gammas

        y0 = gam0 ** (1 / 2)
        y1 = gam1 ** (1 / 2)
        y2 = gam2 ** (1 / 2)

        A0 = [
            [0, y0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, y0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, y0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, y0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]

        A1 = [
            [0, 0, y1, 0, 0, 0, 0, 0],
            [0, 0, 0, y1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, y1, 0],
            [0, 0, 0, 0, 0, 0, 0, y1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]

        A2 = [
            [0, 0, 0, 0, y2, 0, 0, 0],
            [0, 0, 0, 0, 0, y2, 0, 0],
            [0, 0, 0, 0, 0, 0, y2, 0],
            [0, 0, 0, 0, 0, 0, 0, y2],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]

        return np.array([A0, A1, A2])
