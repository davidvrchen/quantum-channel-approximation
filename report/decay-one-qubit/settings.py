import numpy as np
import scipy as sc
import qutip as qt
import matplotlib.pyplot as plt

from q_channel_approx.physics_defns import *
from q_channel_approx.plotting import *
from q_channel_approx import *

m = 1
omegas = (0.4,)
gammas = (0.15,)

ryd_interaction = 0.2

system = DecaySystem(m=m, omegas=omegas, ryd_interaction=ryd_interaction)
