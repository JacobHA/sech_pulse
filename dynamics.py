import numpy as np
from utils import TLS
import matplotlib.pyplot as plt
# Define time points
n_points = 10_000
t_points = np.linspace(-15, 15, n_points)

beta = 4*np.pi
delta = -beta/2
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
sys = TLS('square', beta=beta, delta=delta, alpha=5*np.pi, t_points=t_points)
print(sys.alpha/np.pi)
sys.evolve()
sys.plot()
sys.plot_bloch()

