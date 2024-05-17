import os
from argparse import ArgumentParser

from q_lab.scripts import (
    solve_lindblad,
    make_training_data,
    plot_reference_solution,
    train_circuit,
    plot_approx_channel,
    compare_evolutions,
    optimize_theta,
)
import matplotlib.pyplot as plt


# setup the argument parsing for CLI
parser = ArgumentParser(prog="q-lab")

parser.add_argument(
    "-f", "--folder", type=str, help="enter name of folder to operate on", required=True
)


args = parser.parse_args()


def main() -> None:
    """Qlab script that maps the arguments passed on the command line
    to the appropriate functions to execute the actions in the right place.
    """

    # locate the folder where it all needs to happen
    current_dir = os.getcwd()
    folder_name = args.folder
    path = os.path.join(current_dir, folder_name)

    if not os.path.exists(path):
        print(f"folder {folder_name} does not exist, making new folder")
        os.makedirs(path)

    HELP_MSG = """Available actions to perform: \r
    - solve lindblad,\r
    - plot reference soln, plot reference solution, \r
    - mk training data, make training data, \r
    - train circuit, \r
    - plot approx channel, plot approximated channel, \r
    - compare evolutions
    - q, quit, exit: leave the program
    - h, help, ?: display help message"""

    TERMINATE = ("q", "quit", "exit")

    while (action := input(">>> ")) not in TERMINATE:

        match action:
            case act if act in ("?", "h", "help"):
                print(HELP_MSG)

            case "solve lindblad":
                solve_lindblad(path=path)
                plt.show()

            case act if act in ("plot reference soln", "plot reference solution"):
                plot_reference_solution(path=path)
                plt.show()

            case act if act in ("mk training data", "make training data"):
                make_training_data(path=path)

            case "train circuit":
                train_circuit(path=path)

            case act if act in ("plot approx channel", "plot approximated channel"):
                plot_approx_channel(path=path)
                plt.show()

            case "compare evolutions":
                compare_evolutions(path=path)
                plt.show()

            case "optim theta":
                optimize_theta(path=path)

            case _:
                print("unknown command; type '?' for help")


if __name__ == "__main__":

    main()
