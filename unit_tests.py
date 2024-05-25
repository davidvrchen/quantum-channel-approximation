import doctest
from os import listdir, getcwd
from os.path import isfile, join


if __name__ == "__main__":

    CURRENT_DIR = getcwd()
    PATH = join(CURRENT_DIR, "tests")
    FILES = [f for f in listdir(PATH) if isfile(join(PATH, f))]

    for file in FILES:
        print(f"Testing file: {file}")
        doctest.testfile(join(PATH, file))
        print("done")
