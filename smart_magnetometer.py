import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import multiprocessing as mp
from utils import TLS

# for later reference: https://noisyopt.readthedocs.io/en/latest/

omega_0 = 2
omega = 2.01

def parallel_simulation(beta, temperature):
    times = np.linspace(-10,10,60)
    system = TLS('sech', beta=beta, delta=omega_0 - omega, t_points=times)
    system.evolve(noise_level=temperature)
    fidelity = system.final_fidelity
    # delete system to free memory
    del system
    return fidelity
    
min_beta = -2
max_beta = 2
num_beta_steps = 200
# use scipy built in optimizer to find beta that maximizes fidelity:
from scipy.optimize import minimize_scalar
mins = []
temps = np.logspace(-6, -2, 10)
for temperature in temps:
  def experiment(beta):
      return parallel_simulation(beta, temperature)
  minimize_result = minimize_scalar(lambda beta: experiment(beta),  
                                  method='brent', 
                                  tol=1e-3, 
                                  bracket=(min_beta, max_beta),
                                  options={'maxiter': 150,}
                                )
  mins.append(minimize_result.x)

plt.figure()
plt.title('Sensor Precision vs Temperature')
plt.plot(temps, mins)
plt.xlabel('Temperature')
plt.ylabel(r'Optimal $\beta$')
plt.xscale('log')
plt.savefig('figures/precision_vs_temperature.png')