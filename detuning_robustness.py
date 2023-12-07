import numpy as np
from utils import TLS
import matplotlib.pyplot as plt
# Define time points
n_points = 5000
t_points = np.linspace(-10, 10, n_points)

beta = 0.0#*np.pi
delta_to_expects = {}
for delta in [0, 3, 10]:
    # setup system:
    sys = TLS('sech', beta=0, delta=delta, alpha=2*np.pi, tau=1, t_points=t_points)
    sys.evolve()
    delta_to_expects[delta] = sys.expect

# 3,1 subplots:
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12,8))
# fontsize
plt.rcParams["font.size"] = 22

for (delta, expects), ax in zip(delta_to_expects.items(), axes):
    ax.plot(t_points, expects[0], label=r"$\langle \sigma_x \rangle$", lw=3, color='k')
    ax.plot(t_points, expects[1], label=r"$\langle \sigma_y \rangle$", lw=3, color='r')
    ax.plot(t_points, expects[2], label=r"$\langle \sigma_z \rangle$", lw=3, color='b')
    # add a textbox with the delta value:
    ax.text(0.05, 0.9, r"$\Delta = {}$".format(delta), transform=ax.transAxes, fontsize=20,
        verticalalignment='top')
axes[0].legend(fontsize=22)

axes[2].set_xlabel(r"Time, $t$ [a.u.]", fontsize=22)
plt.tight_layout()
plt.savefig('figures/sech-detuning.png', dpi=300)