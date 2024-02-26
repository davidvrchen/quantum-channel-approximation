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
from scipy.linalg import ishermitian  # Needed for some basic tests

DensityMatrix = np.ndarray


def ket_str(ket: tuple[int]) -> str:
    """Turn tuple of a ket into a pretty printable state.

    Parameters:
    -----------

    ket : (0, 0, 1) represents the ket state |0 0 1>

    >>> ket_str( (0, 0, 1) )
    '|0 0 1>'

    >>> ket_str( (1,) )
    '|1>'
    """

    return f"|{' '.join(str(qubit) for qubit in ket)}>"


def rho_pure_state(ket: tuple[int]) -> DensityMatrix:
    """Create rho for a pure state represented by ``ket``.

    Parameters:
    -----------

    ket : (0, 0, 1) represents the ket state |0 0 1>

    >>> rho_pure_state( (1, 1) )
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 1.]])

    >>> rho_pure_state( (0, 1, 2) )
    Traceback (most recent call last):
    ...
    AssertionError: Not a valid state: |0 1 2>

    >>> ishermitian(rho_pure_state((0, 1, 1, 0)) )
    True

    >>> np.abs(np.trace( rho_pure_state((1, 0, 0, 0)) ) - 1 ) < 10e-6
    True
    """

    assert all(x in (0, 1) for x in ket), f"Not a valid state: {ket_str(ket)}"

    n_qubits = len(ket)

    binary_str = "".join(str(i) for i in ket)
    pos = int(binary_str, 2)

    rho = np.zeros([2**n_qubits, 2**n_qubits])
    rho[pos, pos] = 1

    return rho


def rho_fully_mixed(n_qubits: int) -> DensityMatrix:
    """Create density matrix for a fully mixed state of ``n_qubits`` qubits.

    Parameters:
    -----------

    n_qubits : number of qubits

    >>> rho_fully_mixed(2)
    array([[0.25, 0.  , 0.  , 0.  ],
           [0.  , 0.25, 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.25]])

    >>> ishermitian(rho_fully_mixed( 5 ) )
    True

    >>> np.abs(np.trace( rho_fully_mixed(4) ) - 1) < 10e-6
    True
    """

    return np.eye(2**n_qubits) / 2**n_qubits


def rand_rho_haar(n_qubits: int) -> DensityMatrix:
    """Create density matrix from Haar state for ``n_qubits`` qubits.

    Haar measure is a uniform probability distribution over the Bloch sphere.

    Parameters:
    -----------

    n_qubits : number of qubits

    Reference : https://pennylane.ai/qml/demos/tutorial_haar_measure/

    >>> rand_rho_haar(3).shape
    (8, 8)

    >>> ishermitian(rand_rho_haar(2))
    True

    >>> np.abs(np.trace( rand_rho_haar(5) ) - 1) < 10e-6
    True
    """

    random_ket = qt.rand_ket_haar(dims=[[2**n_qubits], [1]])
    random_ket.dims = [[2] * n_qubits, [2] * n_qubits]
    random_bra = random_ket.dag()

    return (random_ket * random_bra).full()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
