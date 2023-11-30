import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import multiprocessing as mp
from utils import TLS

omega_0 = 2
omega = 3

def parallel_simulation(beta):
    times = np.linspace(-10,10,100)
    system = TLS('sech', beta=beta, delta=omega_0 - omega, t_points=times)
    system.evolve()
    fidelity = system.final_fidelity
    # delete system to free memory
    del system
    return fidelity
    

if __name__ == '__main__':
    min_beta = -0.5
    max_beta = 2.5
    num_beta_steps = 120
    betas = np.linspace(min_beta, max_beta, num_beta_steps)

    # Parallelize the simulation
    pool = mp.Pool()
    fidelity_list = pool.map(parallel_simulation, betas)
    pool.close()
    pool.join()

    fidelity_array = np.array(fidelity_list)
    # plt.figure()
    # One plot with two yaxes:
    fig, ax1 = plt.subplots()

    max_beta = betas[np.argmax(fidelity_array)]
    
    ax1.plot(betas, fidelity_array, 'k', label=f'Fidelity, max at beta={round(max_beta,3)/(2)}')
    # Also plot second derivative of log fidelity:
    log_fid = np.log(fidelity_array)
    fid_susc = -np.gradient(np.gradient(log_fid))
    max_beta = betas[np.argmax(fid_susc)]
    # new yaxis:
    ax2 = ax1.twinx()
    ax2.plot(betas, fid_susc, 'b', label=f'(rescaled) fidelity susc.\n max at beta={round(max_beta,3)/(2)}')

    # aggregate the two legends:
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.set_xlabel('beta')
    plt.title('Fidelity vs. Detuning')
    plt.savefig('figures/magnetic_sensor.png')
    plt.show()
