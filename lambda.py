
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Duration of sech pulse:
tau = 1
# Define parameters for gate:
theta = np.pi/2
phi = -np.pi/2
# tan theta = Ey / Ex
# omega_0 = a_0*np.exp(0.8j*theta)
param_0 = np.sin(theta/2) * np.exp(1j*phi)
param_1 = -np.cos(theta/2)
# omega_1 = np.sqrt(1 - omega_0*np.conj(omega_0))
# sin theta cos phi, sin theta sin phi, cos theta in (y,x,z)
# eta = np.array([np.cos(theta), np.sin(theta)*np.sin(phi), np.sin(theta)*np.cos(phi)])
eta = np.array([np.sin(theta)*np.cos(phi), np.cos(theta), np.sin(theta)*np.sin(phi)])
print(eta)
# Amplitude of sech pulse
alpha = 2*np.pi

# numerically solve 
gamma = 4*np.pi/4
def angle_func(delta):
    return -2*np.pi*delta / (np.pi**2 - delta**2) - np.tan(gamma)
    # return -2*delta / (1 - delta**2) - np.tan(gamma)


# plot the function to see where the root is:
# delta = np.linspace(-10, 10, 100)
# plt.figure()
# plt.plot(delta, angle_func(delta))
# plt.axhline(0, color='k')
# plt.savefig('figures/lambda-func.png')
# find the root of this function:
from scipy.optimize import root
delta_guess = -2
sol = root(angle_func, delta_guess, tol=1e-10)
print('numerical solution', sol.x)
delta = sol.x[0]
# delta = 1/(np.tan(gamma/2)*np.pi)
# delta = n
# p.pi
# delta=-2
print(delta)
# delta = -np.pi / gamma
# delta = 4

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
          delta * (unit_11 + unit_22))

# Initial state |0⟩
psi_0 = qt.basis(3, 0) + qt.basis(3, 1)
psi_0 = psi_0.unit()


# Simulate time evolution
result = qt.mesolve(H=rabi_ham,
                    rho0=psi_0,
                    tlist=t_points,
                    options=qt.Options(store_states=True))

und_x, dx = [], []
und_y, dy = [], []
und_z, dz = [], []

# Get the projector into the 1,2 subspace:
# qutip ketbra:
# qt.ket
# Rotate result.states into bright, dark basis:
# |D> = -Omega1 |0> + Omega0 |1>
# |B> = Omega0* |0> + Omega1* |1>

# 0, 1, E ---> D, B, E
rotate = qt.Qobj(np.array([[-param_1, param_0, 0],
                            [np.conj(param_0), np.conj(param_1), 0],
                            [0, 0, 1]]))

# make it unitary:
rotate = rotate.unit()
dressed_states = [rotate * state for state in result.states]
undressed_states = result.states
# states = result.states
for dressed, undressed in zip(dressed_states, undressed_states):
    # state = qt.Qobj(qt.Qobj(np.array([[result.states[idx][0][0][0]],[result.states[idx][1][0][0]]])).unit())
    undressed = qt.Qobj(undressed[[0,1]]).unit()
    und_x.append(qt.expect(qt.sigmax(), undressed))
    und_y.append(qt.expect(qt.sigmay(), undressed))
    und_z.append(qt.expect(qt.sigmaz(), undressed))
    dressed = qt.Qobj(dressed[[1,2]]).unit()
    dx.append(qt.expect(qt.sigmax(), dressed))
    dy.append(qt.expect(qt.sigmay(), dressed))
    dz.append(qt.expect(qt.sigmaz(), dressed))



plt.figure()
plt.plot(t_points, und_x, label='x')
plt.plot(t_points, und_y, label='y')
plt.plot(t_points, und_z, label='z')
# plt.plot(t_points, result.expect[0], label='x0')
# plt.plot(t_points, result.expect[1], label='y0')
# plt.plot(t_points, result.expect[2], label='z0')
plt.legend()
plt.savefig('figures/lambda.png')
plt.close()
# plt.show()

# Plot on Bloch sphere

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), subplot_kw=dict(projection='3d'))

# Draw a sphere of radius 1:
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax1.quiver(0, 0, 0, *eta, color='k', arrow_length_ratio=0.1)
# plot the surface of the sphere, filled
ax1.plot_wireframe(x, y, z, color="grey", alpha=0.05)
# Draw the z=0 circle (circumference) darker:
# ax1.plot_wireframe(1*np.cos(u), 1*np.sin(u), np.zeros(u.shape), color="grey", alpha=0.15)
# same for the y=0 and x=0 circles:
# ax1.plot_wireframe(1*np.cos(u), np.zeros(u.shape), 1*np.sin(u), color="grey", alpha=0.15)
# ax1.plot_wireframe(np.zeros(u.shape), 1*np.cos(u), 1*np.sin(u), color="grey", alpha=0.15)
# Plot the first Bloch sphere on ax1
ax1.plot(und_x, und_y, und_z, color='b')
# remove the xyz axes
ax1.set_axis_off()
# proper scaling with ratio
ax1.set_aspect('equal')
# label 0 and 1 with kets
ax1.text(0, 0, 1.2, r'$|0\rangle$', fontsize=15)
ax1.text(0, 0, -1.2, r'$|1\rangle$', fontsize=15)

# Plot the second Bloch sphere on ax2
ax2.plot_wireframe(x, y, z, color="grey", alpha=0.05)
# Draw the z=0 circle (circumference) darker:
# ax2.plot_wireframe(1*np.cos(u), 1*np.sin(u), np.zeros(u.shape), color="grey", alpha=0.15)
# same for the y=0 and x=0 circles:
# ax2.plot_wireframe(1*np.cos(u), np.zeros(u.shape), 1*np.sin(u), color="grey", alpha=0.15)
# ax2.plot_wireframe(np.zeros(u.shape), 1*np.cos(u), 1*np.sin(u), color="grey", alpha=0.15)

ax2.plot(dx, dy, dz, color='r')
# remove the xyz axes
ax2.set_axis_off()
# proper scaling with ratio
ax2.set_aspect('equal')
# label 0 and 1 with kets
ax2.text(0, 0, 1.2, r'$|B\rangle$', fontsize=15)
ax2.text(0, 0, -1.2, r'$|E\rangle$', fontsize=15)

# Adjust layout and save the figure:
# squeeze hspace together
fig.subplots_adjust(hspace=0.1)
fig.tight_layout()
plt.savefig('figures/lambda-bloch.png')
plt.show()