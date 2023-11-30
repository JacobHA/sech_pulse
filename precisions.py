import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import multiprocessing as mp
from plot_precision import precision_vs_delta


from utils import TLS
PRECISION_METHOD = 'susc'

phi=3*np.pi
# plt.figure()
def parallel_simulation(beta, delta):
    system = TLS('sech', beta=beta, delta=delta)
    system.evolve()
    # system.plot(title='precisions')
    fidelity = system.final_fidelity
    # delete system to free memory
    del system
    return fidelity
    
if __name__ == '__main__':
    min_beta = -4.5
    max_beta = 4.5
    num_beta_steps = 200
    betas = np.linspace(min_beta, max_beta, num_beta_steps)
    num_delta_steps = 25
    deltas = np.linspace(-2, 2, num_delta_steps)
    precision_list = []
    plt.figure()

    for delta in deltas:
        print(delta)
        with mp.Pool() as pool:
            # fidelity_list = pool.starmap_async(parallel_simulation, zip(betas, [-delta] * len(betas)))#.wait()
            fidelity_list = pool.starmap(parallel_simulation, zip(betas, [delta] * len(betas)))
        
        if PRECISION_METHOD == 'width':
            # find first beta with fidelity above 0.99:
            b0 = np.where(np.array(fidelity_list) > 0.99)[0][0]
            b1 = np.where(np.array(fidelity_list) < 0.99)[0][-1]
            precision = betas[b1] - betas[b0]

        elif PRECISION_METHOD == 'susc':
            # calculate max of second derivative of log fidelity:
            log_fid = np.log(fidelity_list)
            second_deriv = np.gradient(np.gradient(log_fid))
            precision = np.max(second_deriv)

        # make alpha go from 0 to 1 to 0, peaking at midpoint of deltas:
        alpha = 1 - np.abs(delta) / (deltas[-1] - deltas[0])

        plt.plot(betas, fidelity_list, alpha=alpha, color='blue')
        # plt.plot(betas, np.polyval(spline, betas) + 0.99, label='spline')
        plt.xlabel('beta')
        plt.ylabel('Fidelity')
        plt.title(f'Fidelity vs beta, delta={delta}\nPrecision={precision}')
        # plt.legend()

        # Translate to deltas:
        precision_list.append(np.abs(precision))
    plt.savefig(f'figures/fidelity_vs_beta_{PRECISION_METHOD}.png')

    delta_and_precision = np.array([deltas, precision_list])
    np.save(f'data/precision_vs_delta_{PRECISION_METHOD}.npy', delta_and_precision)

    # run the plot_precision.py script to plot the data:
    precision_vs_delta(PRECISION_METHOD)