import numpy as np
from utils import TLS

# Define time points
n_points = 5000
t_points = np.linspace(-15, 15, n_points)

beta = 0.1#*np.pi
delta = -0.11/2 #-beta/2
sys = TLS('sech', beta=beta, delta=delta, alpha=7*np.pi, t_points=t_points)
print(sys.alpha/np.pi)
sys.evolve()
sys.plot()
sys.plot_bloch()

