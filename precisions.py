import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import multiprocessing as mp


tau = 1

raising = qt.Qobj(np.array([[0, 1], [0, 0]]))
lowering = qt.Qobj(np.array([[0, 0], [1, 0]]))

# Define time points
n_points = 600
t_points = np.linspace(-20, 20, n_points)

def sech(t):
    return 1 / np.cosh(t / tau)

def amplitude(t, alpha):
    return alpha / (np.pi * tau) * sech(t / tau)

def detuning(t, beta, delta):
    # Demokov-Kunikew
    return (beta * t + 2 * (delta) * t + beta * (np.log(np.cosh(t)) - np.log(np.cosh(t_points[0])))) / (
            np.pi * tau)

def rabi_ham(t, args, alpha, beta, delta):
    # Hioe rotating frame
    return 1/2 * amplitude(t, alpha) * np.exp(-1j * detuning(t, beta, delta)) * raising + \
           1/2 * amplitude(t, alpha) * np.exp(1j * detuning(t, beta, delta)) * lowering

psi_0 = qt.basis(2, 0)

phi=3*np.pi

def simulate_evolution(beta, delta, plot=False):
    beta *= np.pi
    alpha = (np.sqrt(phi**2 + beta**2) / (np.pi)) * (np.pi)

    def hamiltonian(t, args):
        return rabi_ham(t, args, alpha, beta, delta)

    # Simulate time evolution
    result = qt.mesolve(H=hamiltonian,
                        rho0=psi_0,
                        tlist=t_points,
                        c_ops=[],
                        e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()],
                        options=qt.Options(store_states=True),
                        args={'alpha': alpha, 'beta': beta})

    # Get population of excited state:
    x_expectation = result.expect[0][-1]
    y_expectation = result.expect[1][-1]
    z_expectation = result.expect[2][-1]
    # if plot:
    #     fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    #     axes.plot(t_points, result.expect[0], label=r'$\langle \sigma_x \rangle$')
    #     axes.plot(t_points, result.expect[1], label=r'$\langle \sigma_y \rangle$')
    #     axes.plot(t_points, result.expect[2], label=r'$\langle \sigma_z \rangle$')
    #     axes.legend(loc=0)
    #     axes.set_xlabel('Time')
    #     axes.set_ylabel('Expectation values')
    #     axes.set_title('Sech Pulse Effect')
    #     plt.savefig(f'sech_{beta}.png')
    #     # plt.show()
    fidelity = qt.fidelity(result.states[-1], qt.basis(2, 1))

    return x_expectation, y_expectation, z_expectation, fidelity

def parallel_simulation(beta, plot):
    result_tuple = simulate_evolution(beta, plot)
    return result_tuple

if __name__ == '__main__':
    min_beta = -3
    max_beta = 3
    num_beta_steps = 100
    betas = np.linspace(min_beta, max_beta, num_beta_steps)
    num_delta_steps = 15
    deltas = np.linspace(-2., 2, num_delta_steps)
    precision_list = []
    plt.figure()

    for delta in deltas:
        print(delta)
        # omega_0 = 2 * np.pi
        # omega = 2.05 * np.pi
        with mp.Pool() as pool:
            results = pool.starmap(parallel_simulation, zip(betas, [-delta] * len(betas)))
        x_expectations_list, y_expectations_list, z_expectations_list, fidelity_list = zip(*results)

        # calculate the precision by the width of fidelity at 99%
        # first do a spline interpolation of fidelity
        # then find the roots of the spline
        # then find the width of the roots
        spline = np.polyfit(betas, [f - 0.99 for f in fidelity_list], 6)
        spline_roots = np.roots(spline)
        # find the width of the real roots
        real_spline_roots = [root.real for root in spline_roots if root.imag == 0]
        # find the width of the roots
        try:

            precision = real_spline_roots[-1] - real_spline_roots[0]
        except IndexError:
            precision = 0


        # plot the fidelity and spline:
        # make alpha go from 0 to 1 to 0, peaking at midpoint of deltas:
        alpha = 1 - np.abs(delta) / (deltas[-1] - deltas[0])

        plt.plot(betas, fidelity_list, alpha=alpha, color='blue')
        # plt.plot(betas, np.polyval(spline, betas) + 0.99, label='spline')
        plt.xlabel('beta')
        plt.ylabel('Fidelity')
        plt.title(f'Fidelity vs beta, delta={delta}\nPrecision={precision}')
        # plt.legend()

        # Translate to deltas:
        precision = precision * (deltas[-1] - deltas[0]) / num_delta_steps
        precision_list.append(1/np.abs(precision))
    plt.savefig(f'figures/fidelity_vs_beta_{delta}.png')


    delta_and_precision = np.array([deltas, precision_list])
    np.save('precision_vs_delta.npy', delta_and_precision)

    # Plot precision vs delta
    plt.figure()
    # ignore infinities:
    precision_list = [precision for precision in precision_list if precision != np.inf]
    deltas_list = [deltas for precision in precision_list if precision != np.inf]
    plt.plot(deltas_list, precision_list)
    plt.xlabel('delta')
    plt.ylabel(r'Width at 99% fidelity')
    plt.title('Error vs delta')
    plt.savefig('error.png')
    # save the data:
