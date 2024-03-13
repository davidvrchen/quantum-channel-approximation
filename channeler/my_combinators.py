"""
Some miscellaneous helper functions inspired by functional programming.


References:
    The functional programming course 2IHP0
    taught by Tom Verhoeff

    

Info:
    Created on Mon March 11 2024

    @author: davidvrchen
"""


def split(f, g):
    """Creates a function that takes a single argument
    which applies both f and g to that argument
    and tuples the result together.

    reference:
        The functional programming course 2IHP0
        taught by Tom Verhoeff
    """

    def f_split_g(x):
        """Create the ``split`` of f and g."""
        return (f(x), g(x))

    return f_split_g


def id_func(x):
    """The polymorphic identity function

    reference:
        The functional programming course 2IHP0
        taught by Tom Verhoeff
    """

    return x
