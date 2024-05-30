"""
Various data classes to that represent target systems
Supported target systems:
    Rabi oscillation with decay (DecaySystem),
    transverse field Ising model (TFIMSystem),
    Hamiltonian is the identity (NothingSystem) (mainly used for testing)
"""

from dataclasses import dataclass, KW_ONLY


@dataclass
class TargetSystem:
    """Dataclass that acts as the baseclass
    for target systems.

    Args:
    -----
    m (int): number of qubits

    gammas (tuple[float]): decay rates of the jump operators
    Note: length must equal number of qubits m

    verbose (bool): inform user about data validation
    """

    _: KW_ONLY
    m: int

    gammas: int

    verbose: bool = False

    def __post_init__(self):
        """Check all validation functions.

        Validation function cannot take arguments and should raise
        an error to signal invalid data.
        Name of validation function should start with "validate".
        """

        def all_validations(obj):
            """Create list of all methods that start with "validate"."""
            return [
                getattr(obj, method)
                for method in dir(obj)
                if method.startswith("validate")
            ]

        if self.verbose:
            print("validating settings")

        for method in all_validations(self):
            method()

        if self.verbose:
            print("validation done!")

    def validate_gammas(self):
        """Validate that enough omegas have been provided to model m qubit target system."""
        if self.verbose:
            print("    validating gammas...")

        if self.m != len(self.gammas):
            raise ValueError(
                f"wrong amount of gammas for {self.m} qubit target system: {self.gammas}"
            )


@dataclass
class DecaySystem(TargetSystem):
    """
    Dataclass that defines a decay target system,
    Rabi oscillations on m atoms with decay.

    Args:
    -----
    m (int): number of qubits

    gammas (tuple[float]): decay rates of the jump operators
    Note: length must equal number of qubits m

    verbose (optional: bool): inform user about data validation

    ryd_interaction (float): Rydberg interaction strength
    between the qubits

    omegas (tuple[float]): the Rabi frequency of the qubits
    Note: length must equal number of qubits m

    >>> DecaySystem(ryd_interaction=0.2,
    ...               omegas=(0.2), # not a tuple! expects (0.2,)
    ...               m=1,
    ...               gammas=(3.2,))
    Traceback (most recent call last):
    ...
    TypeError: object of type 'float' has no len()

    >>> DecaySystem(ryd_interaction=0.2,
    ...               omegas=(0.2,), # not enough omegas for m qubit system
    ...               m=2,
    ...               gammas=(3.2, 0.3))
    Traceback (most recent call last):
    ...
    ValueError: wrong amount of omegas for 2 qubit target system: (0.2,)
    """

    ryd_interaction: float

    omegas: tuple[float]

    def validate_omegas(self):
        """Validate that enough omegas have been provided to model m qubit target system."""
        if self.verbose:
            print("    validating omegas...")

        if self.m != len(self.omegas):
            raise ValueError(
                f"wrong amount of omegas for {self.m} qubit target system: {self.omegas}"
            )


@dataclass
class TFIMSystem(TargetSystem):
    """
    Dataclass that defines transverse field Ising model target system.

    Args:
    -----
    m (int): number of qubits

    gammas (tuple[float]): decay rates of the jump operators
    Note: length must equal number of qubits m

    verbose (optional: bool): inform user about data validation

    j_en (float): neighbour-neighbour coupling strength

    h_en (float): Transverse magnetic field strength
    """

    j_en: float
    h_en: float


@dataclass
class NothingSystem(TargetSystem):
    """
    Dataclass that defines the nothing system, i.e. evolution under the identity.

    Args:
    -----
    m (int): number of qubits

    gammas (tuple[float]): decay rates of the jump operators
    Note: length must equal number of qubits m

    verbose (optional: bool): inform user about data validation
    """


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
