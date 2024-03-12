"""
Provides a function to construct the jump operators
from TargetSystemSettings


References:
    original code for decay_examply by @lviss

Info:
    Created on Mon March 11 2024

    @author: davidvrchen
"""

import qutip as qt


from .settings import TargetSystemSettings


JumpOperator = qt.Qobj


def jump_operators(s: TargetSystemSettings) -> list[JumpOperator]:
    """Construct jump operators needed for the Lindbladian from settings.

    >>> jump_operators(TargetSystemSettings(m=2, gammas=(1, 3.2)))[0]
    Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = False
    Qobj data =
    [[0. 1. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 1.]
     [0. 0. 0. 0.]]

    """

    m = s.m  # for easy of notation save the number of qubits

    if m == 1:
        (gam0,) = s.gammas  # comma needed to unpack the tuple
        y0 = gam0 ** (1 / 2)

        # |1> to |0>, |0> to |0
        A0 = [
            [0, y0],
            [0, 0],
        ]

        return [qt.Qobj(A0, dims=[[2], [2]])]

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

        return [
            qt.Qobj(A0, dims=[[2, 2], [2, 2]]),
            qt.Qobj(A1, dims=[[2, 2], [2, 2]]),
        ]

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

        return [
            qt.Qobj(A0, dims=[[2, 2, 2], [2, 2, 2]]),
            qt.Qobj(A1, dims=[[2, 2, 2], [2, 2, 2]]),
            qt.Qobj(A2, dims=[[2, 2, 2], [2, 2, 2]]),
        ]


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
