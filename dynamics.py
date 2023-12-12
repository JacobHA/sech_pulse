import numpy as np
from utils import TLS
import matplotlib.pyplot as plt
import qutip as qt

# Define time points
n_points = 500
t_points = np.linspace(-10, 10, n_points)

delta = 0.5*np.pi 

sys = TLS('sech', t_points=t_points)
sys.evolve(delta=delta, phi=1*np.pi, initial_state=qt.basis(3,0))
sys.plot()
sys.plot_bloch()
# x=qt.basis(3,0) + qt.basis(3,2)
# x=x.unit()

sys.evolve(delta=delta, phi=3*np.pi, initial_state=sys.states[0][-1], global_phase=np.pi)
sys.plot()
sys.plot_bloch()

# print the final expectation values
print(sys.expect_x[-1])
print(sys.expect_y[-1])
print(sys.expect_z[-1])
