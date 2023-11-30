import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import multiprocessing as mp
from utils import TLS

omega_0 = 2
omega = 2.4

def parallel_simulation(beta):
    times = np.linspace(-12,12,200)
    system = TLS('sech', beta=beta, delta=omega_0 - omega, t_points=times)
    system.evolve()
    fidelity = system.final_fidelity
    # delete system to free memory
    del system
    return fidelity
    
# A smarter way to sample beta values would be to
# sample more densely around the maximum fidelity
# and less densely elsewhere.

def smart_beta_sampling(fidelity_list, betas, beta_err):
    # given the known fidelity list and the betas,
    # guess the location of the maximum fidelity:

    max_beta = betas[np.argmax(fidelity_list)]