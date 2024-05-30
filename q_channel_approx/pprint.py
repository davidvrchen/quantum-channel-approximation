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


def comp_labels(m: int) -> list[str]:

    labels = [
        rf"$|{format(i, f'0{m}b')}\rangle \langle{format(i, f'0{m}b')}|$"
        for i in range(2**m)
    ]

    return labels


def indiv_qubit_labels(m: int) -> list[str]:

    labels = [
        rf"$|{format(i, f'0{m}b')}\rangle \langle{format(i, f'0{m}b')}|$"
        for i in range(2**m)
    ]

    return labels
