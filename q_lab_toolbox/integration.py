"""
Provides some function to make ts to integrate Lindbladian on,
should have been part of solve_lindblad branch

Info:
    Created on Wed March 13 2024

    Last update on Thu Apr 4 2024

    @author: davidvrchen
"""

from dataclasses import dataclass

import numpy as np
import qutip as qt


@dataclass
class Ts:
    """Integration settings

    Args:
    -----
    t_max (float): Solve Lindblad from 0 to t_max

    """

    t_max: float


@dataclass
class BasicLinspace(Ts):
    """Use n_steps evenly spaced time steps (aka np.linspace)
    
    Args:
    -----
    t_max (float): Solve Lindblad from 0 to t_max

    n_steps (int): number of intermediate points
    """

    n_steps: int


def basic_ts(s: BasicLinspace):
    """Create ts array of evenly spaced """

    # read parameters from settings
    t_max = s.t_max
    n_steps = s.n_steps

    # create ts
    return np.linspace(0, t_max, n_steps)


def create_ts(s: Ts):
    """Convenience function that creates the appropriate
    Hamiltonian from settings."""

    if isinstance(s, BasicLinspace):
        return basic_ts(s)
