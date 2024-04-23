from argparse import ArgumentParser
import os

from q_lab.scripts import solve_lindblad, make_training_data, plot_reference_solution

import matplotlib.pyplot as plt


# setup the argument parsing for CLI
parser = ArgumentParser(prog="q-lab")

parser.add_argument(
    "-f", "--folder", type=str, help="enter name of folder to operate on", required=True
)

parser.add_argument(
    "-a",
    "--action",
    metavar="OPTION",
    type=str,
    help="action to perform in the specified folder",
    choices=["solve lindblad", "plot reference soln", "mk training data"],
    required=True,
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

    # make settings file
    match args.action:
        case "solve lindblad":
            solve_lindblad(path=path)
            plt.show()

        case "plot reference soln":
            plot_reference_solution(path=path)
            plt.show()

        case "mk training data":
            make_training_data(path=path)


if __name__ == "__main__":

    main()
