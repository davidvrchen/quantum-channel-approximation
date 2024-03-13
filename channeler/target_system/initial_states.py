"""
Helper functions to create initial states to test the algorithm with.
Supported states : Haar states, fully mixed states, pure states.

References:
    https://pennylane.ai/qml/demos/tutorial_haar_measure/
    
Info:
    Created on Thu Feb 23 2024

    @author: davidvrchen
"""

import numpy as np
import qutip as qt

from .settings import (
    Rho0Settings,
    RandHaarSettings,
    PureStateSettings,
    FullyMixedSettings,
)

DensityMatrix = qt.Qobj


def rho_pure_state(s: PureStateSettings) -> DensityMatrix:
    """Create rho for a pure state represented by ``ket``.

    >>> rho_pure_state( PureStateSettings(ket=(1, 1)) )
    Quantum object: dims = [[4], [4]], shape = (4, 4), type = oper, isherm = True
    Qobj data =
    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 1.]]

    >>> rho_pure_state( PureStateSettings(ket=(0, 1, 2)) )
    Traceback (most recent call last):
    ...
    AssertionError: Not a valid state: |0 1 2>
    """

    # read parameters from settings
    ket = s.ket

    # create rho
    n_qubits = len(ket)

    binary_str = "".join(str(i) for i in ket)
    pos = int(binary_str, 2)

    ket = qt.basis(2**n_qubits, pos)

    return qt.Qobj(ket* ket.dag())


def rho_fully_mixed(s: FullyMixedSettings) -> DensityMatrix:
    """Create density matrix for a fully mixed state of ``m`` qubits.


    >>> rho_fully_mixed(FullyMixedSettings(m=2))
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


def rho_rand_haar(s: RandHaarSettings) -> DensityMatrix:
    """Create density matrix from Haar state for ``m`` qubits.

    Haar measure is a uniform probability distribution over the Bloch sphere.

    Reference:
        https://pennylane.ai/qml/demos/tutorial_haar_measure/

    >>> rho_rand_haar(RandHaarSettings(m=3, seed=42)) # doctest:+ELLIPSIS
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


def create_rho0(s: Rho0Settings) -> DensityMatrix:
    """Convenience function that creates the appropriate
    initial state rho0 from settings."""

    if isinstance(s, RandHaarSettings):
        return rho_rand_haar(s)

    if isinstance(s, PureStateSettings):
        return rho_pure_state(s)

    if isinstance(s, FullyMixedSettings):
        return rho_fully_mixed(s)


if __name__ == "__main__":
    import doctest

    MY_FLAG = doctest.register_optionflag("ELLIPSIS")
    doctest.testmod(verbose=True, optionflags=MY_FLAG)
