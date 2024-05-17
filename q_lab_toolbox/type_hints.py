from typing import TypeAlias, Callable, Any
import numpy as np
import qutip as qt

TrainingData: TypeAlias = Any


Hamiltonian: TypeAlias = qt.Qobj | np.ndarray
JumpOperator: TypeAlias = qt.Qobj

DensityMatrix: TypeAlias = qt.Qobj | np.ndarray

Observable: TypeAlias = qt.Qobj | np.ndarray


Theta: TypeAlias = np.ndarray

Unitary: TypeAlias = np.ndarray
UnitaryFactory: TypeAlias = Callable[[Theta], Unitary]


Evolver: TypeAlias = Callable[[DensityMatrix], list[DensityMatrix]]
EvolutionFactory: TypeAlias = Callable[[Theta], Evolver]


Channel: TypeAlias = Callable[[DensityMatrix], DensityMatrix]
ChannelFactory: TypeAlias = Callable[[Theta], Channel]


LossFunction: TypeAlias = Callable[[Theta, TrainingData], float]
