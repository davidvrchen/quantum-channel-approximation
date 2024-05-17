import doctest

# import all files with doctests
from q_lab_toolbox import (
    initial_states,
    jump_operators,
    target_systems,
    readout_operators,
)

# collect all files that need to be tested
all_files = {
    "initial_states": initial_states,
    "jump_operators": jump_operators,
    "target_systems": target_systems,
    "readout_operatrs": readout_operators,
}


def run_tests(files: dict):
    """Tests all modules that are given in files.
    Args:
    -----
    files (dict): (key: value) -> ("name": module)
    """

    for name, module in files.items():
        print(f"testing {name}")

        doctest.testmod(module, verbose=False)


if __name__ == "__main__":

    run_tests(all_files)
