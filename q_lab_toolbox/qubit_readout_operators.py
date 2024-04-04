r"""
Define list of operators that readout the numerically integrated
reference solution created by mesolve.
Each qubit needs 2 readout operators \|0><0\| and \|1><1\|.


Info:
    Created on Wed Mar 13 2024

    @author: davidvrchen
"""

import itertools

import qutip as qt

from .target_systems import TargetSystemSettings

# messy imports needed for testing...
if __name__ == "__main__":
    import os, sys

    module_dir = os.getcwd()
    import_file = f"{module_dir}/my_combinators.py"
    print(import_file, os.getcwd())
    sys.path.append(os.path.dirname(os.path.expanduser(import_file)))
    from q_lab_toolbox.my_combinators import split
else:
    from .my_combinators import split


def read_00_op(tup) -> qt.Qobj:
    """Read state \|0><0\| on ith qubit in m qubit basis

    Args:
        tup (i, m): ith qubit, m qubit basis

    Returns:
        qt.Qobj: readout operator
    """
    i, m = tup
    op_00 = qt.Qobj([[1, 0], [0, 0]])
    return qt.qip.expand_operator(op_00, m, (i,))


def read_11_op(tup) -> qt.Qobj:
    """Read state \|1><1\| on ith qubit in m qubit basis

    Args:
        tup (i, m): ith qubit, m qubit basis

    Returns:
        qt.Qobj: the readout operator
    """

    i, m = tup
    op_11 = qt.Qobj([[0, 0], [0, 1]])
    return qt.qip.expand_operator(op_11, m, (i,))


read_op_pair = split(read_00_op, read_11_op)


def create_readout_individual_qs(s: TargetSystemSettings) -> list[qt.Qobj]:
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

    # read parameters from settings
    m = s.m

    ops = [read_op_pair((i, m)) for i in range(m)]
    return list(itertools.chain.from_iterable(ops))


def b2op(b):
    if b == "0":
        return qt.Qobj([[1, 0], [0, 0]])
    if b == "1":
        return qt.Qobj([[0, 0], [0, 1]])


def str2op(bs):
    return [b2op(b) for b in bs]


def str2tensor(bs):
    return qt.tensor(str2op(bs))


def create_readout_computational_basis(s: TargetSystemSettings) -> list[qt.Qobj]:
    """_summary_

    Args:
        s (TargetSystemSettings): _description_

    Returns:
        list[qt.Qobj]: _description_
    """

    # read parameters from settings
    m = s.m

    comp_basis = range(2**m)

    ops = [str2tensor(format(bs, f"0{m}b")) for bs in comp_basis]
    return ops


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
