import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import multiprocessing as mp
from utils import TLS

omega_0 = 2
omega = 2.08

def parallel_simulation(beta):
    system = TLS('sech', beta=beta, delta=omega_0 - omega)
    system.evolve()
    fidelity = system.final_fidelity
    # delete system to free memory
    del system
    return fidelity
    

if __name__ == '__main__':
    min_beta = -1
    max_beta = 1
    num_beta_steps = 120
    betas = np.linspace(min_beta, max_beta, num_beta_steps)

    # Parallelize the simulation
    pool = mp.Pool()
    fidelity_list = pool.map(parallel_simulation, betas)
    pool.close()
    pool.join()

    fidelity_array = np.array(fidelity_list)
    plt.figure()
    max_beta = betas[np.argmax(fidelity_array)]
    
    plt.plot(betas, fidelity_array, label=f'Fidelity, max at beta={round(max_beta,3)/(2)}')
    plt.legend()
    plt.title('Fidelity vs. Detuning')
    plt.savefig('fidelity_old.png')
    # plt.show()
