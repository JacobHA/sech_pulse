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
    system.evolve(thermal_temp=0.01)
    # system.plot(title='precisions')
    fidelity = system.final_fidelity
    # delete system to free memory
    del system
    return fidelity
    
if __name__ == '__main__':
    min_beta = -1.5
    max_beta = 4.5
    num_beta_steps = 350
    betas = np.linspace(min_beta, max_beta, num_beta_steps)
    num_delta_steps = 3
    deltas = np.linspace(0,-1.5, num_delta_steps)
    print(-deltas/2)
    precision_list = []
    # plt.figure()
    fig, axes = plt.subplots(num_delta_steps, 1, figsize=(10, 6))
    # plt.rcParams["font.size"] = 22

    for ax, delta in zip(axes, deltas):
        print(delta)
        with mp.Pool() as pool:
            # fidelity_list = pool.starmap_async(parallel_simulation, zip(betas, [-delta] * len(betas)))#.wait()
            fidelity_list = pool.starmap(parallel_simulation, zip(betas, [delta] * len(betas)))
        
        if PRECISION_METHOD == 'width':
            # find first beta with fidelity below 0.01:
            b0 = np.where(np.array(fidelity_list) > 0.01)[0][0]
            b1 = np.where(np.array(fidelity_list) < 0.01)[0][-1]
            precision = betas[b1] - betas[b0]

        elif PRECISION_METHOD == 'susc':
            # calculate max of second derivative of log fidelity:
            log_fid = np.log(fidelity_list)
            second_deriv = np.gradient(np.gradient(log_fid))
            precision = np.max(second_deriv)
            fid_derivative = np.gradient(fidelity_list, 1)
            precision = (max(np.diff(fid_derivative)))

        # make alpha go from 0 to 1 to 0, peaking at midpoint of deltas:
        # alpha = 1 - np.abs(delta) / (deltas[-1] - deltas[0])
        ax.plot(betas, fidelity_list, '-', label=f'Fidelity', color='blue', lw=3)#, alpha=alpha, color='blue')
        ax.set_ylabel('Fidelity', fontsize=16)

        # Plot susceptibility on new y axis:
        ax2 = ax.twinx()
        log_fid = np.log(fidelity_list)
        ax2.plot(betas, -np.gradient(log_fid,2), '-', label='Susceptibility', color='orange', lw=3)
        # show legend: of both axes in same legend:
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # add a legend saying delta={delta}
        # title the legend with delta={delta}
        ax.legend(lines + lines2, labels + labels2, loc='upper right', title=r'$\Delta$ = {:.2f}'.format(-2*delta))
        # make the legend opaque:
        ax2.set_ylabel('Susceptibility', fontsize=16)
        # plt.plot(betas, np.polyval(spline, betas) + 0.99, label='spline')
        ax.get_legend().get_frame().set_facecolor('white')


        # Translate to deltas:
        precision_list.append(np.abs(precision))
    # ax.set_ylabel('Fidelity')
    # ax.set_ylabel('Susceptibility')
    ax.set_xlabel(r'Laser Chirp $\beta$', fontsize=18)
    axes[0].set_title(r'Fidelity Transitions', fontsize=24)
    plt.tight_layout()

    plt.savefig(f'figures/fidelity_vs_beta_{PRECISION_METHOD}.png')

    delta_and_precision = np.array([deltas, precision_list])
    np.save(f'data/precision_vs_delta_{PRECISION_METHOD}.npy', delta_and_precision)

    # run the plot_precision.py script to plot the data:
    precision_vs_delta(PRECISION_METHOD)