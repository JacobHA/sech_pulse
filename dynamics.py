import numpy as np
from utils import TLS

# Define time points
n_points = 1000
t_points = np.linspace(-10, 10, n_points)

beta = 1.4*np.pi
sys = TLS('square', beta=beta, delta=-beta/2, phi=3*np.pi, t_points=t_points)
print(sys.alpha/np.pi)
sys.evolve()
sys.plot()
sys.plot_bloch()

