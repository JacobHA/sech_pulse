import numpy as np
from utils import TLS
import matplotlib.pyplot as plt
import qutip as qt

# Define time points
n_points = 500
t_points = np.linspace(-10, 10, n_points)

deltas = np.linspace(-5, 5, 160)
# Make a bloch sphere:
b = qt.Bloch()
plt.figure()
thetas = []
delta_to_theta = {}
def experiment(delta):
    sys = TLS('sech', t_points=t_points)
    sys.evolve(delta=delta, phi=1*np.pi, initial_state=qt.basis(3,0), thermal_temp=0.01)
    sys.evolve(delta=delta, phi=3*np.pi, initial_state=sys.states[0][-1], global_phase=np.pi, thermal_temp=0.01)

    # get y,z final expectation values:
    x = sys.expect_x[-1]
    y = sys.expect_y[-1]
    z = sys.expect_z[-1]
    # get angle:

    theta = np.pi + np.arctan2(x, z)
    print(theta)
    theta = x/z
    thetas.append(theta)
    # delta_to_theta[delta] = theta
    return theta

import multiprocessing as mp
pool = mp.Pool()
with pool:
    thetas = pool.map(experiment, deltas)
    for delta, theta in zip(deltas, thetas):
        delta_to_theta[delta] = theta
pool.close()
pool.join()

# for delta in deltas:
#     experiment(delta)

plt.figure()
print(delta_to_theta)
plt.plot(delta_to_theta.keys(), delta_to_theta.values())
plt.xlabel(r'$\delta$')
plt.ylabel('Rotation angle with |0>')
plt.savefig('figures/yxz-overlap.png')

b.save('figures/bloch_overlap.png')