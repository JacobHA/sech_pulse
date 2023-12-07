from utils import TLS
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

# Define time points
n_points = 500
delta = 0.5
excited = qt.basis(2, 1)
plt.figure()
drives = np.linspace(0, 5, 100)
max_pops = []
for drive in drives:
    # Calculate a pi pulse, A=1
    T = 6*np.pi
    t_points = np.linspace(0, T, n_points)

    sys = TLS('rabi', beta=drive, delta=delta, phi=7*np.pi, t_points=t_points)
    sys.evolve()
    states = sys.states
    # get population of excited state via overlap
    pops = [qt.fidelity(state,excited)**2 for state in states[0]]
    # pops = sys.expect[2]**2

    # plt.plot(t_points, pops, label=rf'$\beta={drive}$')
    max_pops.append(1-max(pops))

# Plot max pop vs drives:
plt.plot(drives, max_pops, label=rf'$\delta={delta}$')


plt.legend()
plt.show()
plt.savefig('figures/rabi-error.png')