# Plot the data at precision_vs_delta.npy
import numpy as np
import matplotlib.pyplot as plt



def precision_vs_delta(method):

    deltas, precision_list = np.load(f'data/precision_vs_delta_{method}.npy')

    plt.figure()
    # ignore infinities:
    # precision_list = [precision for precision in precision_list if precision != np.inf]
    # deltas = [deltas for precision in precision_list if precision != np.inf]
    plt.plot(-deltas, precision_list)
    plt.xlabel(r'Detunings, $\Delta$')
    plt.ylabel(r'Width at 1% fidelity')
    plt.title('Error vs delta')
    plt.savefig('error.png')
    plt.savefig('figures/precision_vs_delta.png')
    plt.show()

if __name__ == '__main__':
    PRECISION_METHOD = 'susc'
    precision_vs_delta(PRECISION_METHOD)