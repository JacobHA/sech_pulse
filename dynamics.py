import numpy as np
from utils import LTS

# Define time points
n_points = 1000
t_points = np.linspace(-10, 10, n_points)

beta = 1*np.pi
sys = LTS('sech', beta=beta, delta=-beta/2, phi=4*np.pi, t_points=t_points)
print(sys.alpha/np.pi)
sys.evolve()
sys.plot()
sys.plot_bloch()

