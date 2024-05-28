import doctest
import q_lab_toolbox.physics_defns.initial_states as qlab


if __name__ == "__main__":
    print("testing")
    MY_FLAG = doctest.register_optionflag("ELLIPSIS")
    doctest.testmod(qlab, verbose=True, optionflags=MY_FLAG)
    print("done")
