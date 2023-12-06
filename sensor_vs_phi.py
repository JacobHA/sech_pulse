import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import multiprocessing as mp
from utils import TLS

omega_0 = 0
omega = 1.8

import scipy.optimize as opt
def power_law(x, a, b, c):
    return a * (x - delta)**b + c

def parallel_simulation(beta, phi):
    times = np.linspace(-25,25,500)
    system = TLS('sech', phi=phi*np.pi, beta=beta, delta=omega_0 - omega, t_points=times)
    system.evolve()
    fidelity = system.final_fidelity
    # delete system to free memory
    del system
    return fidelity
    
B = -2*(omega_0 - omega_0)
if __name__ == '__main__':
    min_beta = 12#0.2*0.99
    max_beta = -4# 0.2*1.01
    num_beta_steps = 100
    betas = np.linspace(min_beta, max_beta, num_beta_steps)
    beta_err = np.abs(betas[1] - betas[0])
    fidelity_arrays = {}
    for phi in [1,3,7]:
        def experiment(beta):
            return parallel_simulation(beta, phi)
        # Parallelize the simulation
        pool = mp.Pool(20)
        fidelity_list = pool.map(experiment, betas)
        pool.close()
        pool.join()

        fidelity_arrays[phi] = np.array(fidelity_list)

    fig, ax1 = plt.subplots()
    # Draw a vertical line at the true magnetic field:
    ax1.axvline(x=-(omega_0 - omega), color='r', linestyle='--', label=rf'True $B={-round(omega_0 - omega, 2)}$')
    # plot for all phis, with different colors in a gradient from blue to red:
    # set cmap
    cmap = plt.get_cmap('viridis')
    for phi, fidelity_array in fidelity_arrays.items():
        ax1.plot(betas/2, fidelity_array, label=rf'$\phi={phi}\pi$', color=cmap(phi/13))
    # legend on right, outside of plot:
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncols=1)

    # Adjust layout
    fig.subplots_adjust(bottom=0.4)

    ax1.set_ylabel('Fidelity With Initial State')
    ax1.set_xlabel(r'Chirp Magnitude, $\beta$')
    plt.title('Fidelity vs. Detuning')
    plt.tight_layout()
    plt.savefig(f'figures/magnetic_sensor.png')
    # save the data:
    np.save(f'data/magnetic_sensor.npy', fidelity_arrays)