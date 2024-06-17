import numpy as np
import scipy as sc
import qutip as qt
import matplotlib.pyplot as plt

from q_channel_approx.physics_defns import *
from q_channel_approx.plotting import *
from q_channel_approx import *

m = 2
gammas = (0.2, 0.15)


system = NothingSystem(m=m)
