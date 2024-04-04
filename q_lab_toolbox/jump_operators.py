"""
Provides a functions to construct the jump operators
from TargetSystem object


References:
    original code for decay_examply by @lviss

Info:
    Created on Mon March 11 2024

    Last update on Thu Apr 4 2024

    @author: davidvrchen
"""

import qutip as qt


if __name__ == "__main__":
    from target_systems import TargetSystem, DecaySystem, TFIMSystem
else:
    from .target_systems import TargetSystem, DecaySystem, TFIMSystem


def _default_jump_operators(m: int, gammas: tuple[float]):
    """Construct the default jump operators as defined in original code by @lviss.

    >>> _default_jump_operators( m=2, gammas=(1, 3.2) )[0]
    Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = False
    Qobj data =
    [[0. 1. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 1.]
     [0. 0. 0. 0.]]

    """

    if m == 1:
        (gam0,) = gammas  # comma needed to unpack the tuple
        y0 = gam0 ** (1 / 2)

        # |1> to |0>, |0> to |0
        A0 = [
            [0, y0],
            [0, 0],
        ]

        return [qt.Qobj(A0, dims=[[2], [2]])]

    if m == 2:
        gam0, gam1 = gammas

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

        return [
            qt.Qobj(A0, dims=[[2, 2], [2, 2]]),
            qt.Qobj(A1, dims=[[2, 2], [2, 2]]),
        ]

    if m == 3:
        gam0, gam1, gam2 = gammas

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

        return [
            qt.Qobj(A0, dims=[[2, 2, 2], [2, 2, 2]]),
            qt.Qobj(A1, dims=[[2, 2, 2], [2, 2, 2]]),
            qt.Qobj(A2, dims=[[2, 2, 2], [2, 2, 2]]),
        ]


def default_jump_operators(s: DecaySystem | TFIMSystem):
    """Convenience function to create the default jump operators from
    DecaySystem or TFIMSystem objects.
    """
    # read the settings
    m = s.m
    gammas = s.gammas

    return _default_jump_operators(m=m, gammas=gammas)


def create_jump_operators(s: TargetSystem):
    """Convenienec function that created the appropriate jump operators
    for the provided subclass of TargetSystem object."""

    if isinstance(s, DecaySystem) or isinstance(S, TFIMSystem):
        return default_jump_operators(s)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
