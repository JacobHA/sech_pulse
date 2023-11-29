from matplotlib import pyplot as plt
import numpy as np
import qutip as qt

raising = qt.Qobj(np.array([[0, 1], [0, 0]]))
lowering = qt.Qobj(np.array([[0, 0], [1, 0]]))
psi_0 = qt.basis(2, 0)
psi_1 = qt.basis(2, 1)
omega_0 = 2*np.pi
omega = 2.*np.pi


def sech(t):
    return 1/np.cosh(t)

def sech_amplitude(t, alpha, tau=1):
    return alpha/(np.pi * tau) * sech(t/tau)

def sech_detuning(t, beta, delta=0, tau=1, t0=-10):#Demokov-Kunikew
    return (beta*t + 2*delta*t + beta*(np.log(np.cosh(t))- np.log(np.cosh(t0))))/(np.pi * tau)#(omega_0 - omega)*t


def alpha_func(phi, beta):
    return (np.sqrt(phi**2 + beta**2)/(np.pi))*(np.pi)

# def sech(t):
#     return 1 / np.cosh(t)

def square_amplitude(t, alpha, tau=1):
    if t < -1 / (np.pi*tau):
        return 0
    elif t > 1 / (np.pi*tau):
        return 0
    else:
        return alpha / (np.pi * tau)

def square_detuning(t, beta, delta, tau=1):
    if t < -1*(np.pi*tau)/2:
        return 0
    elif t > (np.pi*tau)/2:
        return 0
    else:
        return -((beta+2*delta)*tau*(np.arctan(np.sin(-np.pi/2)- np.arctan(np.sin(t/tau))))) + beta*tau*np.log(np.cos(-np.pi/2)*sech(t/tau))

class TLS:
    def __init__(self, pulse_shape, psi_0=None, t_points=None, ham=None, alpha=None, beta=None, delta=None, phi=None, tau=1):
        self.psi_0 = psi_0 if psi_0 is not None else qt.basis(2, 0)
        self.t_points = t_points if t_points is not None else np.linspace(-10, 10, 100)
        self.t0 = self.t_points[0]
        self.delta = delta if delta is not None else 0
        self.phi = phi if phi is not None else np.pi
        self.beta = beta if beta is not None else beta
        self.alpha = alpha if alpha is not None else alpha_func(self.phi, beta)
        self.tau = tau
        print(self.phi**2 - (self.alpha**2 - self.beta**2))
        self.states = None
        self.expect = None
        self.evolved = False

        if pulse_shape == 'square':
            self.amplitude = lambda t: square_amplitude(t, self.alpha, tau=self.tau)
            self.detuning = lambda t: square_detuning(t, self.beta, self.delta, tau=self.tau)
        elif pulse_shape == 'sech':
            self.amplitude = lambda t: sech_amplitude(t, self.alpha, tau=self.tau)
            self.detuning = lambda t: sech_detuning(t, self.beta, delta=self.delta, tau=self.tau, t0=self.t0)
        else:
            raise ValueError('pulse_shape must be "square" or "sech"')
        

    def ham(self,t,args):#Hioe rotating frame
        return 1/2*self.amplitude(t) * \
                (np.exp(-1j*self.detuning(t))*raising + \
                 np.exp(1j*self.detuning(t))*lowering)


    def evolve(self,):
        result = qt.mesolve(H=self.ham,
                            rho0=self.psi_0,
                            tlist=self.t_points,
                            c_ops=[],
                            e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()],
                            options=qt.Options(store_states=True))
        self.states = result.states
        self.expect = result.expect
        self.fidelities = qt.fidelity(result.states[-1], psi_1)
        self.evolved = True

        return 
    
    def plot(self,title='dynamics'):
        assert self.evolved, "System not yet evolved. Nothing to plot."
        # two subplots, one with expects, one with pulse
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        ax1.plot(self.t_points, self.expect[0], label=r'$\langle \sigma_x \rangle$')
        ax1.plot(self.t_points, self.expect[1], label=r'$\langle \sigma_y \rangle$')
        ax1.plot(self.t_points, self.expect[2], label=r'$\langle \sigma_z \rangle$')
        ax1.legend(loc=0)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Expectation values')
        ax1.set_title('Sech Pulse Effect')

        ax2.plot(self.t_points, [self.amplitude(t_point) for t_point in self.t_points])
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Pulse amplitude')
        ax2.set_title('Pulse amplitude')
        plt.savefig(title + '.png')
        return
    
    def plot_bloch(self, title='bloch-path'):
        assert self.evolved, "System not yet evolved. Nothing to plot."

        # Plot on Bloch sphere
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='3d'))

        ax.axis('square') # to get a nice circular plot
        ax.set_title('Bloch Sphere')
        b = qt.Bloch(fig=fig, axes=ax)
        b.add_points([self.expect[0], self.expect[1], self.expect[2]], meth='l')

        fig.tight_layout()
        b.save(title + '.png')