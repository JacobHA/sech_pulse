import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import multiprocessing as mp

precision = 20
range_val = 1

omega_0 = 2 * np.pi
omega = 2.05 * np.pi
tau = 1

raising = qt.Qobj(np.array([[0, 1], [0, 0]]))
lowering = qt.Qobj(np.array([[0, 0], [1, 0]]))

# Define time points
n_points = 200
t_points = np.linspace(-10, 10, n_points)

def sech(t):
    return 1 / np.cosh(t / tau)

def amplitude(t, alpha):
    return alpha / (np.pi * tau) * sech(t / tau)

def detuning(t, beta):
    # Demokov-Kunikew
    return (beta * t + 2 * (omega_0 - omega) * t + beta * (np.log(np.cosh(t)) - np.log(np.cosh(t_points[0])))) / (
            np.pi * tau)

def rabi_ham(t, args, alpha, beta):
    # Hioe rotating frame
    return 1/2 * amplitude(t, alpha) * np.exp(-1j * detuning(t, beta)) * raising + \
           1/2 * amplitude(t, alpha) * np.exp(1j * detuning(t, beta)) * lowering

psi_0 = qt.basis(2, 0)


def simulate_evolution(beta, plot=False):
    beta *= np.pi
    alpha = (np.sqrt(np.pi * np.pi + beta * beta) / (np.pi)) * (np.pi)
    print('alpha', alpha)
    print('beta', beta)

    def hamiltonian(t, args):
        return rabi_ham(t, args, alpha, beta)

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
    if plot:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes.plot(t_points, result.expect[0], label=r'$\langle \sigma_x \rangle$')
        axes.plot(t_points, result.expect[1], label=r'$\langle \sigma_y \rangle$')
        axes.plot(t_points, result.expect[2], label=r'$\langle \sigma_z \rangle$')
        axes.legend(loc=0)
        axes.set_xlabel('Time')
        axes.set_ylabel('Expectation values')
        axes.set_title('Sech Pulse Effect')
        plt.savefig(f'sech_{beta}.png')
        # plt.show()
    fidelity = qt.fidelity(result.states[-1], qt.basis(2, 1))
    print(fidelity)

    return x_expectation, y_expectation, z_expectation, fidelity

def parallel_simulation(beta, plot):
    result_tuple = simulate_evolution(beta, plot)
    return result_tuple

if __name__ == '__main__':
    min_beta = -1
    max_beta = 2
    num_beta_steps = 60
    betas = np.linspace(min_beta, max_beta, num_beta_steps)

    #plot only if beta is first or last
    plots = [True if beta == min_beta or beta == max_beta else False for beta in betas]
    with mp.Pool() as pool:
        results = pool.starmap(parallel_simulation, zip(betas, plots))
    x_expectations_list, y_expectations_list, z_expectations_list, fidelity_list = zip(*results)

    x_expectations_array = np.array(x_expectations_list)
    y_expectations_array = np.array(y_expectations_list)
    z_expectations_array = np.array(z_expectations_list)
    fidelity_array = np.array(fidelity_list)
    plt.figure()
    max_beta = betas[np.argmax(fidelity_array)]
    
    plt.plot(betas, fidelity_array, label=f'Fidelity, max at beta={round(max_beta,2)}')
    plt.legend()
    plt.title('Fidelity vs. Detuning')
    plt.savefig('fidelity_old.png')
    # plt.show()
