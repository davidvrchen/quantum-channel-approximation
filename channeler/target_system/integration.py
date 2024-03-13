"""
Provides some function to make ts to integrate Lindbladian on.

Info:
    Created on Wed March 13 2024

    @author: davidvrchen
"""

import numpy as np
import qutip as qt

from .settings import TsSettings, BasicLinspace


def basic_ts(s: BasicLinspace):

    # read parameters from settings
    t_max = s.t_max
    n_steps = s.n_steps

    # create ts
    return np.linspace(0, t_max, n_steps)


def create_ts(s: TsSettings):
    """Convenience function that creates the appropriate
    Hamiltonian from settings."""

    if isinstance(s, BasicLinspace):
        return basic_ts(s)


