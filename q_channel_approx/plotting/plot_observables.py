"""
Defines `create_observables_comp_basis` function
along with some helper functions to make it work.
"""

import qutip as qt


def char2obs(char: str) -> qt.Qobj:
    """Returns the one qubit observable corresponding to
    an observation of char in computational basis

    Args:
        char (str): one qubit state in computational basis

    Returns:
        qt.Qobj: the one qubit observable
    """

    if char == "0":
        return qt.Qobj(
            [
                [1, 0],
                [0, 0],
            ]
        )

    if char == "1":
        return qt.Qobj(
            [
                [0, 0],
                [0, 1],
            ]
        )

    raise ValueError(f"{char} cannot be converted to one qubit observable")


def str2obs(s: str) -> list[qt.Qobj]:
    """Create a list of one qubit observables from a string
    encoding of the one qubit observables

    Args:
        s (str): encoding of single qubit observables

    Returns:
        list[qt.Qobj]: list of one qubit observables
    """

    return [char2obs(b) for b in s]


def str2tensor(s: str) -> qt.Qobj:
    """Create the observable corresponding to `s` where
    `s` is a string encoding of a state in the single qubit basis.

    Args:
        s (str): encoding of single qubit observables

    Returns:
        qt.Qobj: the observables corresponding to `s`
    """

    return qt.tensor(str2obs(s))


def create_observables_comp_basis(m: int) -> list[qt.Qobj]:
    """Create all observables on `m` qubits (in the single qubit basis).

    Args:
        m (int): number of qubits.

    Returns:
        list[qt.Qobj]: list of observables in the single qubit basis.
    """

    comp_basis = range(2**m)

    ops = [str2tensor(f"{bs:0{m}b}") for bs in comp_basis]
    return ops
