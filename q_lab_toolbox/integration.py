"""
Provides some function to make ts to integrate Lindbladian on.

Info:
    Created on Wed March 13 2024

    @author: davidvrchen
"""

from dataclasses import dataclass

import numpy as np
import qutip as qt

from .target_systems import TsSettings, BasicLinspace


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


@dataclass
class TsSettings:
    """Integration settings

    Args:

    t_max (float): Solve Lindblad from 0 to t_max

    """

    t_max: float


@dataclass
class BasicLinspace(TsSettings):
    """"""

    n_steps: int
