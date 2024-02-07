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
    times = np.linspace(-15, 15, 400)
    system = TLS('sech', beta=beta, delta=delta, t_points=times)
    system.evolve(thermal_temp=0.0)
    # system.plot(title='precisions')
    # Get final expectation value of Z
    fidelity = system.expect[-1][-1]
    # delete system to free memory
    del system
    return fidelity
    
if __name__ == '__main__':
    min_beta = -1.5
    max_beta = 4.5
    num_beta_steps = 50
    betas = np.linspace(min_beta, max_beta, num_beta_steps)
    num_delta_steps = 1
    deltas = np.array([-0.2])
    print(-deltas/2)
    precision_list = []
    # plt.figure()
    fig, axes = plt.subplots(num_delta_steps, 1, figsize=(10, 6))
    # plt.rcParams["font.size"] = 22

    for ax, delta in zip([axes], deltas):
        print(delta)
        with mp.Pool() as pool:
            # fidelity_list = pool.starmap_async(parallel_simulation, zip(betas, [-delta] * len(betas)))#.wait()
            fidelity_list = pool.starmap(parallel_simulation, zip(betas, [delta] * len(betas)))
        
        # make alpha go from 0 to 1 to 0, peaking at midpoint of deltas:
        # alpha = 1 - np.abs(delta) / (deltas[-1] - deltas[0])
        ax.plot(betas, fidelity_list, '-', label=f'Fidelity', color='blue', lw=3)#, alpha=alpha, color='blue')
        ax.set_ylabel(r'$\langle \sigma_z | \Psi | \sigma_z \rangle$', fontsize=16)

        # show legend: of both axes in same legend:
        lines, labels = ax.get_legend_handles_labels()
        # add a legend saying delta={delta}
        # title the legend with delta={delta}
        ax.legend(lines, labels, loc='upper right', title=r'$\gamma$ = {:.2f}'.format(-2*delta),
                   title_fontsize='16', fontsize='16')
        # make the legend opaque:
        ax.hlines(-1, min_beta, max_beta, color='black', linestyle='--', label='Min. Expectation')
        # plt.plot(betas, np.polyval(spline, betas) + 0.99, label='spline')
        ax.get_legend().get_frame().set_facecolor('white')


    ax.set_xlabel(r'Laser Chirp $\beta$', fontsize=18)
    ax.set_title(r'$\sigma_{z}$ Expectation', fontsize=24)
    # plt.tight_layout()

    # Inrease fontsizes on axes:
    axes.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(f'figures/expect_vs_beta.png')

    # delta_and_precision = np.array([deltas, precision_list])
    # np.save(f'data/precision_vs_delta_{PRECISION_METHOD}.npy', delta_and_precision)

    # run the plot_precision.py script to plot the data:
    precision_vs_delta(PRECISION_METHOD)