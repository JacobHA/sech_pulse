
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
# Duration of sech pulse
tau = 1
# Define parameters
# Energy Gap = Driving Frequency (?)
theta = np.pi/2
phi = -np.pi/2
# tan theta = Ey / Ex
# omega_0 = a_0*np.exp(0.8j*theta)
param_0 = np.sin(theta/2) * np.exp(1j*phi)
param_1 = -np.cos(theta/2)
# omega_1 = np.sqrt(1 - omega_0*np.conj(omega_0))
# sin theta cos phi, sin theta sin phi, cos theta in (y,x,z)
eta = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
print(eta)
# Amplitude of sech pulse
alpha = 2*np.pi
assert alpha == 2*np.pi
# phi = 2*np.pi
beta = 0#np.sqrt(alpha**2 - phi**2)
# delta = 1/(tau*np.tan(gamma/2))
# numerically solve 
gamma = np.pi
def angle_func(delta):
    return -2*np.pi*delta / (np.pi**2 - delta**2) - np.tan(gamma/2)

# find the root of this function:
from scipy.optimize import root
delta_guess = 0.1
sol = root(angle_func, delta_guess)
print(sol.x)
delta = sol.x[0]


unit_11 = qt.Qobj(np.array([[1,0,0],[0,0,0],[0,0,0]]))
unit_13 = qt.Qobj(np.array([[0,0,1],[0,0,0],[0,0,0]]))
unit_22 = qt.Qobj(np.array([[0,0,0],[0,1,0],[0,0,0]]))
unit_23 = qt.Qobj(np.array([[0,0,0],[0,0,1],[0,0,0]]))
unit_31 = qt.Qobj(np.array([[0,0,0],[0,0,0],[1,0,0]]))
unit_32 = qt.Qobj(np.array([[0,0,0],[0,0,0],[0,1,0]]))
unit_33 = qt.Qobj(np.array([[0,0,0],[0,0,0],[0,0,1]]))


# Define time points
n_points = 500
t_points = np.linspace(-15, 15, n_points)

def sech(t):
    return 1/np.cosh(t/tau)

def Amplitude(t):
    return alpha/(np.pi * tau) * sech(t/tau)

def rabi_ham(t, args):
    return -1/2*(Amplitude(t)*(param_0*unit_13 + np.conj(param_0)*unit_31 + param_1*unit_23 + np.conj(param_1)*unit_32) +\
          (delta + beta * np.tanh(t / tau)) * (unit_11 + unit_22))

        # unit_22 * (beta * np.tanh(t / tau)) / (np.pi * tau) 

# ensure hamiltonian is Hermitian
# assert(rabi_ham(1,1).isherm), "Hamiltonian is not Hermitian"

# Initial state |0⟩
psi_0 = qt.basis(3, 0) #+ qt.basis(3, 1)
psi_0 = psi_0.unit()

# make an operator which is sigmaz padded with zeros in the lower right:
x_op = qt.Qobj(np.array([[0,1,0],
                         [1,0,0],
                         [0,0,0]])
    )
y_op = qt.Qobj(np.array([[0,-1j,0],
                         [1j,0,0],
                         [0,0,0]])
        )
z_op = qt.Qobj(np.array([[1,0,0],
                         [0,-1,0],
                         [0,0,0]])
)
# Simulate time evolution
result = qt.mesolve(H=rabi_ham,
                    rho0=psi_0,
                    tlist=t_points,
                    c_ops=[],
                    e_ops=[x_op, y_op, z_op],
                    options=qt.Options(store_states=True))

x_exps = []
y_exps = []
z_exps = []

# Get the projector into the 1,2 subspace:
# qutip ketbra:
# qt.ket
for idx, t in enumerate(t_points):
    state = qt.Qobj(qt.Qobj(np.array([[result.states[idx][0][0][0]],[result.states[idx][1][0][0]]])).unit())
    x_exps.append(qt.expect(qt.sigmax(), state))
    y_exps.append(qt.expect(qt.sigmay(), state))
    z_exps.append(qt.expect(qt.sigmaz(), state))

plt.figure()
plt.plot(t_points, x_exps, label='x')
plt.plot(t_points, y_exps, label='y')
plt.plot(t_points, z_exps, label='z')
# plt.plot(t_points, result.expect[0], label='x0')
# plt.plot(t_points, result.expect[1], label='y0')
# plt.plot(t_points, result.expect[2], label='z0')
plt.legend()
plt.savefig('figures/lambda.png')
# plt.show()

# Plot on Bloch sphere
fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='3d'))

ax.axis('square') # to get a nice circular plot
ax.set_title('Bloch Sphere')
b = qt.Bloch(fig=fig, axes=ax)
b.add_vectors(eta)

b.add_points([x_exps, y_exps, z_exps], meth='l')
# get the rotation vector:
# sin theta cos phi, sin theta sin phi, cos theta in (y,x,z)

# swap x->y
# eta[0], eta[1] = eta[1], eta[0]
# Plot on Bloch sphere
b.show()
fig.tight_layout()
b.save(f'figures/lambda-bloch.png')