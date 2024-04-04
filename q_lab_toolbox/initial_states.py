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


def rho_pure_state(s: RhoPureState):
    """Create rho for a pure state represented by ``ket``.

    >>> rho_pure_state( RhoPureState(m=2, ket=(1, 1)) )
    Quantum object: dims = [[4], [4]], shape = (4, 4), type = oper, isherm = True
    Qobj data =
    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 1.]]

    >>> rho_pure_state( RhoPureState(m=3, ket=(0, 1, 2)) )
    Traceback (most recent call last):
    ...
    AssertionError: Not a valid ket in qubit basis: |0 1 2>
    """

    # read parameters from settings
    ket = s.ket

    # create rho
    n_qubits = len(ket)

    binary_str = "".join(str(i) for i in ket)
    pos = int(binary_str, 2)

    ket = qt.basis(2**n_qubits, pos)

    return qt.Qobj(ket * ket.dag())


def rho_fully_mixed(s: RhoFullyMixed):
    """Create density matrix for a fully mixed state of ``m`` qubits.


    >>> rho_fully_mixed(RhoFullyMixed(m=2))
    Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
    Qobj data =
    [[0.25 0.   0.   0.  ]
     [0.   0.25 0.   0.  ]
     [0.   0.   0.25 0.  ]
     [0.   0.   0.   0.25]]
    """
    # read parameters from settings
    m = s.m

    # create rho
    return qt.Qobj(np.eye(2**m) / 2**m, dims=[[2] * m, [2] * m])


def rho_rand_haar(s: RhoRandHaar):
    """Create density matrix from Haar state for ``m`` qubits.

    Haar measure is a uniform probability distribution over the Bloch sphere.

    Reference:
        https://pennylane.ai/qml/demos/tutorial_haar_measure/

    >>> rho_rand_haar(RhoRandHaar(m=3, seed=42)) # doctest:+ELLIPSIS
    Quantum object: dims = [[2, 2, 2], [2, 2, 2]], shape = (8, 8), type = oper, isherm = True
    Qobj data =
    ...
    """
    # read parameters from settings
    seed = s.seed
    m = s.m

    # create rho
    random_ket = qt.rand_ket_haar(dims=[[2] * m, [1] * m], seed=seed)
    random_bra = random_ket.dag()

    return random_ket * random_bra


def create_rho0(s: Rho0):
    """Convenience function that creates the appropriate
    initial state rho0 from settings."""

    if isinstance(s, RhoRandHaar):
        return rho_rand_haar(s)

    if isinstance(s, RhoPureState):
        return rho_pure_state(s)

    if isinstance(s, RhoFullyMixed):
        return rho_fully_mixed(s)


if __name__ == "__main__":
    import doctest

    MY_FLAG = doctest.register_optionflag("ELLIPSIS")
    doctest.testmod(verbose=True, optionflags=MY_FLAG)
