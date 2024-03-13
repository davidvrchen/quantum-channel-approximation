"""
THIS FILE IS DEPRECIATED

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

DensityMatrix = qt.Qobj


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


def rho_pure_state(ket: tuple[int]) -> DensityMatrix:
    """Create rho for a pure state represented by ``ket``.

    Parameters:
    -----------

    ket : (0, 0, 1) represents the ket state \|0 0 1>

    >>> rho_pure_state( (1, 1) )
    Quantum object: dims = [[4], [4]], shape = (4, 4), type = oper, isherm = True
    Qobj data =
    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 1.]]

    >>> rho_pure_state( (0, 1, 2) )
    Traceback (most recent call last):
    ...
    AssertionError: Not a valid state: |0 1 2>
    """

    assert all(x in (0, 1) for x in ket), f"Not a valid state: {ket_str(ket)}"

    n_qubits = len(ket)

    binary_str = "".join(str(i) for i in ket)
    pos = int(binary_str, 2)

    rho = np.zeros([2**n_qubits, 2**n_qubits])
    rho[pos, pos] = 1

    return qt.Qobj(rho)


def rho_fully_mixed(m: int) -> DensityMatrix:
    """Create density matrix for a fully mixed state of ``m`` qubits.

    Parameters:
    -----------

    m: number of qubits

    >>> rho_fully_mixed(2)
    Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
    Qobj data =
    [[0.25 0.   0.   0.  ]
     [0.   0.25 0.   0.  ]
     [0.   0.   0.25 0.  ]
     [0.   0.   0.   0.25]]
    """

    return qt.Qobj(np.eye(2**m) / 2**m, dims=[[2] * m, [2] * m])


def rand_rho_haar(m: int) -> DensityMatrix:
    """Create density matrix from Haar state for ``m`` qubits.

    Haar measure is a uniform probability distribution over the Bloch sphere.

    Parameters:
    -----------

    m: number of qubits

    Reference:
        https://pennylane.ai/qml/demos/tutorial_haar_measure/

    >>> rand_rho_haar(3) # doctest:+ELLIPSIS
    Quantum object: dims = [[2, 2, 2], [2, 2, 2]], shape = (8, 8), type = oper, isherm = True
    Qobj data =
    ...
    """

    random_ket = qt.rand_ket_haar(dims=[[2] * m, [1] * m], seed=42)
    random_bra = random_ket.dag()

    return random_ket * random_bra


if __name__ == "__main__":
    import doctest

    MY_FLAG = doctest.register_optionflag("ELLIPSIS")
    doctest.testmod(verbose=True, optionflags=MY_FLAG)
