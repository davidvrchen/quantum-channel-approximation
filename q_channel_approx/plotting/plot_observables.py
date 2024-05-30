import qutip as qt
from itertools import chain


def read_00_op(tup) -> qt.Qobj:
    """Read state \|0><0\| on ith qubit in m qubit basis

    Args:
        tup (i, m): ith qubit, m qubit basis

    Returns:
        qt.Qobj: readout operator
    """
    i, m = tup
    op_00 = qt.Qobj([[1, 0], [0, 0]])
    return qt.expand_operator(op_00, m, (i,))


def read_11_op(tup) -> qt.Qobj:
    """Read state \|1><1\| on ith qubit in m qubit basis

    Args:
        tup (i, m): ith qubit, m qubit basis

    Returns:
        qt.Qobj: the readout operator
    """

    i, m = tup
    op_11 = qt.Qobj([[0, 0], [0, 1]])
    return qt.expand_operator(op_11, m, (i,))


read_op_pair = lambda x: (read_00_op(x), read_11_op(x))


def create_observables_individual_qs(m: int) -> list[qt.Qobj]:
    """Create list of readout operators for each qubit in
    m qubit basis.
    Each qubit is read out as \|0><0\| and \|1><1\|

    Args:
        s (TargetSystemSettings): settings

    >>> create_readout_ops( TargetSystemSettings(m=1, gammas=(0,) ))[0]
    Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
    Qobj data =
    [[1. 0.]
     [0. 0.]]

    >>> create_readout_ops( TargetSystemSettings(m=1, gammas=(0,) ))[1]
    Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
    Qobj data =
    [[0. 0.]
     [0. 1.]]
    """

    ops = [read_op_pair((i, m)) for i in range(m)]
    return list(chain.from_iterable(ops))


def b2op(b) -> qt.Qobj:
    if b == "0":
        return qt.Qobj([[1, 0], [0, 0]])
    if b == "1":
        return qt.Qobj([[0, 0], [0, 1]])


def str2op(bs) -> list[qt.Qobj]:
    return [b2op(b) for b in bs]


def str2tensor(bs) -> qt.Qobj:
    return qt.tensor(str2op(bs))


def create_observables_comp_basis(m: int) -> list[qt.Qobj]:
    """to be added"""

    comp_basis = range(2**m)

    ops = [str2tensor(format(bs, f"0{m}b")) for bs in comp_basis]
    return ops


if __name__ == "__main__":
    import doctest

    MY_FLAG = doctest.register_optionflag("ELLIPSIS")
    doctest.testmod(verbose=True, optionflags=MY_FLAG)
