import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import multiprocessing as mp
from utils import TLS

omega_0 = 2
omega = 2.05

import scipy.optimize as opt
def power_law(x, a, b, c):
    return a * (x - delta)**b + c

def parallel_simulation(beta):
    times = np.linspace(-20,20,2000)
    system = TLS('sech', phi=3*np.pi, beta=beta, delta=omega_0 - omega, t_points=times)
    system.evolve()
    fidelity = system.final_fidelity
    # delete system to free memory
    del system
    return fidelity
    

if __name__ == '__main__':
    min_beta = -1
    max_beta = 1
    num_beta_steps = 300
    betas = np.linspace(min_beta, max_beta, num_beta_steps)
    beta_err = np.abs(betas[1] - betas[0])

    # Parallelize the simulation
    pool = mp.Pool(20)
    fidelity_list = pool.map(parallel_simulation, betas)
    pool.close()
    pool.join()

    fidelity_array = np.array(fidelity_list)
    # calculate critical exponent by fitting a power law near the minimum fidelity (0):
    # which should occur at the detuning, delta = omega_0 - omega
    # delta = omega_0 - omega
    # # find the index of the minimum fidelity:
    # min_fid_index = np.argmin(fidelity_array)
    # right_fidelity_array = fidelity_array[min_fid_index:]
    # right_betas = betas[min_fid_index:]
    # # fit a power law:
    # right_popt, _ = opt.curve_fit(power_law, right_betas/2, right_fidelity_array)
    # print(right_popt[1])
    # # now for left side:
    # left_fidelity_array = fidelity_array[:min_fid_index]
    # left_betas = betas[:min_fid_index]
    # # fit a power law:
    # left_popt, _ = opt.curve_fit(power_law, -left_betas/2, left_fidelity_array)
    # print(left_popt[1])
    
    # # plt.figure()
    # # One plot with two yaxes:

    fig, ax1 = plt.subplots()
    # Draw a vertical line at the true magnetic field:
    ax1.axvline(x=-(omega_0 - omega), color='r', linestyle='--', label=rf'True Magnetic Field: $-2\delta={-round(omega_0 - omega, 2)}$')

    max_beta = betas[np.argmin(fidelity_array)]

    ax1.plot(betas/2, fidelity_array, 'k', label=rf'Fidelity: Min. at $\beta={round(max_beta,3)/(2)}\pm{round(beta_err,3)/(2)}$')

    log_fid = np.log(fidelity_array)
    fid_derivative = np.gradient(fidelity_array, 1)
    print(max(np.diff(fid_derivative)))
    print(rf'Occured at\n $\beta={betas[np.argmax(np.diff(fid_derivative))]/2}$')
    fid_susc = -np.gradient(log_fid, 2)
    # Average across the PT
    fid_susc_beta = (betas[np.argmax(fid_susc)] + betas[np.argmin(fid_susc)])/2

    ax2 = ax1.twinx()
    ax2.plot(betas/2, fid_susc, 'b', label=rf'Fidelity Susceptibility: P.T. at $\beta={round(max_beta,3)/(2)}$')

    # Combine handles and labels from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Create a single legend with multiline labels
    legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)

    # Adjust layout
    fig.subplots_adjust(bottom=0.4)

    ax1.set_ylabel('Fidelity With Initial State')
    ax1.set_xlabel(r'Chirp Magnitude, $\beta$')
    plt.title('Fidelity vs. Detuning')
    plt.savefig(f'figures/magnetic_sensor.png')