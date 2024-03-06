# quantum-channel-approximation
Python code for simulating quantum channels using quantum computers.

The required python packages can be installed with the requirements.txt file (note the specific numpy version).
To install all packages at once you can use
> python -m pip install -r requirements.txt

Lindbladian_via_stinespring is the main file that can be run to use the full program. This file is set up in various blocks:
- Setting up parameters
    parameters are for setting the original lindbladian to apprimate, 
    defining the type of circuit used to make the approximation, 
    and setting the parameters for training the circuit.
- Initialize the class
- Define the evolution operator of the lindbladian (1-3 qubits, TFIM or simple decay with rabi oscillations and rydberg interaction)
- Manually set the initial training data and plot the exact solution of the lindbladian
- Train the unitary in full via class functions
- Reapply the unitary via class functions to investigate the quality of the approximation.

The files have internal documentation per function and class.
These is also some html documentation available (open the file docs/_build/html/index.html)
The html documentation can be rendered using sphinx.

The Example_{} files are copies of Lindbladian_via_stinespring with some example settings and can be run directly.
