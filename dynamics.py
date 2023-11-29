import numpy as np
from utils import TLS

# Define time points
n_points = 1000
t_points = np.linspace(-10, 10, n_points)

beta = 3*np.pi
sys = TLS('sech', beta=beta, delta=-beta/2)
sys.evolve()
sys.plot()
sys.plot_bloch()

# get population of excited state:

