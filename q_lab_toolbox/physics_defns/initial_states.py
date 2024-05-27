"""
Various data classes to that represent initial states and some 
helper functions to create them.
Supported states: Haar states, fully mixed states, pure states.

References:
    https://pennylane.ai/qml/demos/tutorial_haar_measure/

Info:
    Created on Thu Feb 23 2024

    Last update on Thu Apr 4 2024

    @author: davidvrchen
"""

from dataclasses import dataclass
import random

import numpy as np
import qutip as qt


@dataclass
class Rho0:
    """Dataclass that acts as baseclass
    for initial states rho0.

    Args:
    -----

    m (int): number of qubits, allows 2**n dimensional state
    """

    m: int


@dataclass
class RhoRandHaar(Rho0):
    """Dataclass that defines an initial rho generated from a ket
    generated from a Haar state.

    Args:
    -----
    m (int): number of qubits, allows 2**n dimensional state

    seed (int, optional): seed passed to qt.rand_ket_haar.
    If None provided, a random seed between [0, 1000] is used.
    """

    seed: int = None

    def __post_init__(self):
        """If no seed is provided, set the seed to be a random number between 0 and 1000."""
        if self.seed is None:
            random_seed = random.randint(0, 1000)
            print(
                f"No seed supplied to RhoRandHaar\nUsing randomly generated seed {random_seed}"
            )
            self.seed = random_seed


@dataclass
class RhoFullyMixed(Rho0):
    """Dataclass that defines a fully mixed initial state.

    Args:
    -----
    m (int): number of qubits, allows 2**n dimensional state
    """


@dataclass
class RhoPureState(Rho0):
    """Dataclass that defines an initial state generated
    from a pure state.
    tuple        ket        density matrix
    (0, 0, 1) -> |0 0 1> -> |0 0 1> <0 0 1|

    Args:
    -----
    m (int): number of qubits, allows 2**n dimensional state

    ket (tuple): representation of ket used to make density matrix

    >>> RhoPureState(m=2, ket=(0, 1))
    RhoPureState(m=2, ket=(0, 1))

    >>> RhoPureState(m=1, ket=(1, 0))
    Traceback (most recent call last):
    ...
    AssertionError: Ket |1 0> is not a state on 1 qubits

    >>> RhoPureState(m=2, ket=(0, 2))
    Traceback (most recent call last):
    ...
    AssertionError: Not a valid ket in qubit basis: |0 2>
    """

    ket: tuple

    def __post_init__(self):
        """Checks if the ket is a valid ket."""
        assert all(
            x in (0, 1) for x in self.ket
        ), f"Not a valid ket in qubit basis: {ket_str(self.ket)}"
        assert (
            len(self.ket) == self.m
        ), f"Ket {ket_str(self.ket)} is not a state on {self.m} qubits"


def ket_str(ket: tuple[int]) -> str:
    """Turn tuple of a ket into a pretty printable state.

    Parameters:
    -----------

    ket : (0, 0, 1) represents the ket state \|0 0 1>

    >>> ket_str( (0, 0, 1) )
    '|0 0 1>'

    >>> ket_str( (1,) )
    '|1>'
    """

    return f"|{' '.join(str(qubit) for qubit in ket)}>"


def rho_pure_state(ket: tuple):
    """Create rho for a pure state represented by ``ket``.

    >>> _rho_pure_state( ket=(1, 1) )
    Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
    Qobj data =
    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 1.]]
    """

    # create rho
    m = len(ket)

    binary_str = "".join(str(i) for i in ket)
    pos = int(binary_str, 2)

    ket = qt.basis(2**m, pos)

    return qt.Qobj(ket * ket.dag(), dims=[[2]*m, [2]*m])


def rho_fully_mixed(m: int):
    """Create density matrix for a fully mixed state of ``m`` qubits.


    >>> _rho_fully_mixed(m=2)
    Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
    Qobj data =
    [[0.25 0.   0.   0.  ]
     [0.   0.25 0.   0.  ]
     [0.   0.   0.25 0.  ]
     [0.   0.   0.   0.25]]
    """

    # create rho
    return qt.Qobj(np.eye(2**m) / 2**m, dims=[[2] * m, [2] * m])


def rho_rand_haar(m: int, seed: int):
    """Create density matrix from Haar state for ``m`` qubits.

    Haar measure is a uniform probability distribution over the Bloch sphere.

    Reference:
        https://pennylane.ai/qml/demos/tutorial_haar_measure/

    >>> _rho_rand_haar( m=3, seed=42 ) # doctest:+ELLIPSIS
    Quantum object: dims = [[2, 2, 2], [2, 2, 2]], shape = (8, 8), type = oper, isherm = True
    Qobj data =
    ...
    """
    # create rho
    random_ket = qt.rand_ket_haar(dims=[[2] * m, [1] * m], seed=seed)
    random_bra = random_ket.dag()

    return random_ket * random_bra


def _rho_rand_haar(s: RhoRandHaar):
    """Convenience function to create rho from RhoRandHaar object."""
    # read parameters from settings
    seed = s.seed
    m = s.m

    return rho_rand_haar(m=m, seed=seed)


def _rho_fully_mixed(s: RhoFullyMixed):
    """Convenience function to create rho from RhoFullyMixed object."""
    # read parameters from settings
    m = s.m

    # create rho
    return rho_fully_mixed(m=m)


def _rho_pure_state(s: RhoPureState):
    """Convenience function to create rho from RhoPureState object."""
    # read parameters from settings
    ket = s.ket

    return rho_pure_state(ket=ket)


def create_rho0(s: Rho0):
    """Convenience function that creates the appropriate
    initial state rho0 from settings."""

    if isinstance(s, RhoRandHaar):
        return _rho_rand_haar(s)

    if isinstance(s, RhoPureState):
        return _rho_pure_state(s)

    if isinstance(s, RhoFullyMixed):
        return _rho_fully_mixed(s)


if __name__ == "__main__":
    import doctest

    MY_FLAG = doctest.register_optionflag("ELLIPSIS")
    doctest.testmod(verbose=True, optionflags=MY_FLAG)
