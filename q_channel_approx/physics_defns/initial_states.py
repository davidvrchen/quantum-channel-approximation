"""
Various functions to create initial states (density matrices)
Supported states: Haar states, fully mixed states, pure states.
"""

import numpy as np
import qutip as qt


def rho_pure_state(ket: tuple) -> qt.Qobj:
    """Create rho for a pure state represented by ``ket``.

    >>> rho_pure_state( ket=(1, 1) )
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
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

    return qt.Qobj(ket * ket.dag(), dims=[[2] * m, [2] * m])


def rho_fully_mixed(m: int) -> qt.Qobj:
    """Create density matrix for a fully mixed state of ``m`` qubits.


    >>> rho_fully_mixed(m=2)
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[0.25 0.   0.   0.  ]
     [0.   0.25 0.   0.  ]
     [0.   0.   0.25 0.  ]
     [0.   0.   0.   0.25]]
    """

    # create rho
    return qt.Qobj(np.eye(2**m) / 2**m, dims=[[2] * m, [2] * m])


def rho_rand_haar(m: int, seed: int = None) -> qt.Qobj:
    """Create density matrix from Haar state for ``m`` qubits.

    Haar measure is a uniform probability distribution over the Bloch sphere.

    Reference:
        https://pennylane.ai/qml/demos/tutorial_haar_measure/

    >>> rho_rand_haar( m=3, seed=42 ) # doctest:+ELLIPSIS
    Quantum object: dims=[[2, 2, 2], [2, 2, 2]], shape=(8, 8), type='oper', dtype=Dense, isherm=True
    Qobj data =
    ...
    """
    if seed is None:
        seed = np.random.default_rng().integers(10**5)
        print(f"rho_rand_haar: {seed=}")
    # create rho
    random_ket = qt.rand_ket(
        dimensions=[[2] * m, [1] * m], seed=seed, distribution="haar"
    )
    random_bra = random_ket.dag()

    rho = random_ket * random_bra
    rho.dims = [[2] * m, [2] * m]

    return rho


if __name__ == "__main__":
    import doctest

    MY_FLAG = doctest.register_optionflag("ELLIPSIS")
    doctest.testmod(verbose=True, optionflags=MY_FLAG)
