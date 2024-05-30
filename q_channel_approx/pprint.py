"""
Various functions related to displaying output in a prettier way.
"""

from typing import Iterable


def ket2str(ket: Iterable[int]) -> str:
    """Turn a ket represented as a tuple (in qubit basis)
    into a pretty printable string

    Args:
        ket (iterable[int]): representation of a ket as an iterable

    Returns:
        str: pretty printable string

    >>> ket2str( (0, 0, 1) )
    '|0 0 1>'

    >>> ket2str( (1,) )
    '|1>'

    >>> ket2str( (0, 1, 2) )
    Traceback (most recent call last):
    ...
    AssertionError: ket=(0, 1, 2) not given in qubit basis
    """

    assert all(qubit in (0, 1) for qubit in ket), f"{ket=} not given in qubit basis"

    return f"|{' '.join(str(qubit) for qubit in ket)}>"


def comp_basis_labels(m: int) -> list[str]:
    """Return the labels of states in the computational basis.

    Args:
        m (int): number of qubits.

    Returns:
        list[str]: list of labels of each state in the computational basis.
    """

    labels = [rf"$|{i}\rangle \langle{i}|$" for i in range(2**m)]

    return labels


def indiv_q_basis_labels(m: int) -> list[str]:
    """Return the labels of states in the individual qubit basis.

    Args:
        m (int): number of qubits.

    Returns:
        list[str]: list of labels of each state in the individual qubit basis.
    """

    labels = [rf"$|{i:0{m}b}\rangle \langle{i:0{m}b}|$" for i in range(2**m)]

    return labels
