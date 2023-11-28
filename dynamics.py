import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import multiprocessing as mp

precision = 20
range = 1

beta_list = []
x_expectations_list = []
y_expectations_list = []
z_expectations_list = []
fidelity_list = []
omega_0 = 2*np.pi
omega = 2.01*np.pi
tau = 1

raising = qt.Qobj(np.array([[0,1],[0,0]]))
lowering = qt.Qobj(np.array([[0,0],[1,0]]))

# Define time points
n_points = 100
t_points = np.linspace(-10, 10, n_points)

def sech(t):
    return 1/np.cosh(t/tau)

def Amplitude(t):
    return alpha/(np.pi * tau) * sech(t/tau)

def Detuning(t):#Demokov-Kunikew
    return (beta*t + 2*(omega_0 - omega)*t + beta*(np.log(np.cosh(t))- np.log(np.cosh(t_points[0]))))/(np.pi * tau)#(omega_0 - omega)*t
#def Detuning(t):#Allen-Ebberly
    return ((omega_0 - omega)*t + beta*(np.log(np.cosh(t))- np.log(np.cosh(t_points[0]))))/(np.pi * tau)#(omega_0 - omega)*t

def rabi_ham(t, args):#Hioe rotating frame
    return 1/2*Amplitude(t)*np.exp(-1j*Detuning(t))*raising + 1/2*Amplitude(t)*np.exp(1j*Detuning(t))*lowering


psi_0 = qt.basis(2,0)
min_beta = -1
max_beta = 2
num_beta_steps = 40
betas = np.linspace(min_beta,max_beta,num_beta_steps)

for beta in betas:
    beta *= np.pi
    alpha = (np.sqrt(np.pi*np.pi + beta*beta)/(np.pi))*(np.pi)


    # Simulate time evolution
    result = qt.mesolve(H=rabi_ham,
                        rho0=psi_0,
                        tlist=t_points,
                        c_ops=[],
                        e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()],
                        options=qt.Options(store_states=True))

    # get population of excited state:

    x_expectations_list.append(result.expect[0][-1])
    y_expectations_list.append(result.expect[1][-1])
    z_expectations_list.append(result.expect[2][-1])
    fidelity_list.append(qt.fidelity(result.states[-1],qt.basis(2,1)))
    print(fidelity_list[-1])

x_expectations_array = np.array(x_expectations_list)
y_expectations_array = np.array(y_expectations_list)
z_expectations_array = np.array(z_expectations_list)
fidelity_array = np.array((fidelity_list))

#print(alpha*alpha - beta*beta)
#print(np.pi*np.pi)
#print(beta/np.pi)
#print(result.expect[0][-1],result.expect[1][-1],result.expect[2][-1])

plt.plot(betas,fidelity_array)
#plt.plot(beta_array/2,x_expectations_array,beta_array/2,y_expectations_array)
#plt.plot(beta_array/2,z_expectations_array)
plt.savefig('fidelity_old.png')
# plt.show()