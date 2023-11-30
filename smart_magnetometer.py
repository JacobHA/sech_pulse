import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import multiprocessing as mp
from utils import TLS

# for later reference: https://noisyopt.readthedocs.io/en/latest/

omega_0 = 2
omega = 2.01

def parallel_simulation(beta):
    times = np.linspace(-10,10,60)
    system = TLS('sech', beta=beta, delta=omega_0 - omega, t_points=times)
    system.evolve()
    fidelity = system.final_fidelity
    # delete system to free memory
    del system
    return fidelity
    
# use scipy built in optimizer to find beta that maximizes fidelity:
from scipy.optimize import minimize_scalar
min_beta = -2
max_beta = 2
num_beta_steps = 200
minimize_result = minimize_scalar(lambda beta: parallel_simulation(beta),  
                                  method='brent', 
                                  tol=1e-3, 
                                  bracket=(min_beta, max_beta),
                                  options={'maxiter': 20,}
                                    
                                )

print(minimize_result)
