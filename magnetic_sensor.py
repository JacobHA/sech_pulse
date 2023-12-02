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
    times = np.linspace(-20,20,300)
    system = TLS('sech', beta=beta, delta=omega_0 - omega, t_points=times)
    system.evolve()
    fidelity = system.final_fidelity
    # delete system to free memory
    del system
    return fidelity
    

if __name__ == '__main__':
    min_beta = -4
    max_beta = 6
    num_beta_steps = 700
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

    max_beta = betas[np.argmax(fidelity_array)]
    
    ax1.plot(betas/2, fidelity_array, 'k', label=f'Fidelity, max at beta={round(max_beta,3)/(2)}+/-{round(beta_err,3)/(2)}')
    # # plot the fit:
    # ax1.plot(left_betas/2, power_law(-left_betas/2, *left_popt), 'r--', label=f'Power law fit, exponent={round(left_popt[1],3)}')
    # ax1.plot(right_betas/2, power_law(right_betas/2, *right_popt), 'r--', label=f'Power law fit, exponent={round(right_popt[1],3)}')
    # # Also plot second derivative of log fidelity:
    log_fid = np.log(fidelity_array)
    fid_derivative = np.gradient(fidelity_array, 1)
    print(max(np.diff(fid_derivative)))
    print(f'Occured at beta={betas[np.argmax(np.diff(fid_derivative))]/2}')
    fid_susc = -np.gradient(log_fid, 2)
    max_beta = betas[np.argmax(fid_susc)]
    # new yaxis:
    ax2 = ax1.twinx()
    ax2.plot(betas/2, fid_susc, 'b', label=f'Fidelity susc.\n max at beta={round(max_beta,3)/(2)}')

    # aggregate the two legends:
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    ax1.set_xlabel('beta')
    plt.title('Fidelity vs. Detuning')
    plt.savefig('figures/magnetic_sensor.png')
    plt.show()
