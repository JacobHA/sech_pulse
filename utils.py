from matplotlib import pyplot as plt
import numpy as np
import qutip as qt

# want to drive both transitions:
slot_02 = qt.Qobj(np.array([[0, 0, 1], 
                            [0, 0, 0], 
                            [0, 0, 0]]))

slot_20 = qt.Qobj(np.array([[0, 0, 0], 
                            [0, 0, 0], 
                            [1, 0, 0]]))

slot_12 = qt.Qobj(np.array([[0, 0, 0], 
                            [0, 0, 1], 
                            [0, 0, 0]]))

slot_21 = qt.Qobj(np.array([[0, 0, 0], 
                            [0, 0, 0], 
                            [0, 1, 0]]))

slot_11 = qt.Qobj(np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]]))

slot_22 = qt.Qobj(np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 1]]))

psi_0 = qt.basis(3, 0)
psi_1 = qt.basis(3, 1)
omega_0 = 2*np.pi
omega = 2.*np.pi


def sech(t): 
    return 1/np.cosh(t)

def sech_amplitude(t, alpha, tau=1):
    return alpha/(np.pi * tau) * sech(t/tau)

def sech_detuning(t, beta, delta=0, tau=1, t0=-10):#Demokov-Kunikew
    return (beta*t + 2*delta*t + beta*(np.log(np.cosh(t)) - np.log(np.cosh(t0))))/(np.pi * tau)#(omega_0 - omega)*t


def alpha_func(phi, beta):
    return (np.sqrt(phi**2 + beta**2))

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
        return -((beta+2*delta)*tau*(np.arctan(np.sin(-np.pi/2) - np.arctan(np.sin(t/tau))))) + beta*tau*np.log(np.cos(-np.pi/2)*sech(t/tau))

class LTS:
    def __init__(self, pulse_shape, psi_0=None, t_points=None, ham=None, alpha=None, beta=None, delta=0, phi=np.pi, tau=1):
        self.pulse_name = pulse_shape
        self.psi_0 = psi_0 if psi_0 is not None else qt.basis(3, 0)
        self.t_points = t_points if t_points is not None else np.linspace(-10, 10, 100)
        self.t0 = self.t_points[0]
        self.delta = delta
        self.phi = phi
        self.beta = beta
        self.alpha = alpha if alpha is not None else alpha_func(self.phi, beta)
        self.tau = tau
        self.states = None
        self.expect = None
        self.final_fidelity = None
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
        Delta=0.
        return 1/2*self.amplitude(t) * \
                (np.exp(-1j*self.detuning(t))*slot_02 + \
                 np.exp(1j*self.detuning(t))*slot_20 )#+ \
                
                # np.exp(-1j*self.detuning(t))*slot_12 + \
                # np.exp(1j*self.detuning(t))*slot_21
                # ) + \
                # -2*Delta * slot_22

                # 2*(deltaw - w1) * slot_11 + \ # set to zero (\delta=0)
    

    def evolve(self,):
        # qt.smesolve
        result = qt.mesolve(H=self.ham,
                            rho0=self.psi_0,
                            tlist=self.t_points,
                            c_ops=[],
                            # e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()],
                            options=qt.Options(store_states=True))
        # use smesolve to get the states and expectation values:
        # convert self.ham to qobj:
        # qham = qt.Qobj(self.ham, dims=[[2], [2]])#, input_dims=[[2], [2]])
        # result = qt.smesolve(H=qham,
        #                     times=self.t_points,
        #                     rho0=self.psi_0,
        #                     c_ops=[],
        #                     e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()],
        #                     options=qt.Options(store_states=True))
        self.states = result.states
        self.expect = result.expect
        self.final_fidelity = qt.fidelity(result.states[-1], psi_0)
        self.evolved = True

        return 
    
    def plot_fidelity(self, title='fidelity'):
        assert self.evolved, "System not yet evolved. Nothing to plot."
        plt.figure()

        states = [qt.Qobj(state[:-1]) for state in self.states]
        # renormalize states:
        states = [state.unit() for state in states]
        psi0 = qt.basis(2, 0)
        plt.plot(self.t_points, [qt.fidelity(state, psi0) for state in states])
        plt.xlabel('Time')
        plt.ylabel('Fidelity')
        plt.title('Fidelity vs time')
        plt.savefig(f'figures/{self.pulse_name}-{title}.png')
        return

    def plot_dyn(self,title='dynamics'):
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
        # put some vspace between the two plots
        plt.subplots_adjust(hspace=0.5)


        ax2.plot(self.t_points, [self.amplitude(t_point) for t_point in self.t_points])
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Pulse amplitude')
        ax2.set_title('Pulse amplitude')
        plt.savefig(f'figures/{self.pulse_name}-{title}.png')
        return
    
    def plot_bloch(self, title='bloch-path'):
        assert self.evolved, "System not yet evolved. Nothing to plot."

        # Plot on Bloch sphere
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='3d'))

        ax.axis('square') # to get a nice circular plot
        ax.set_title('Bloch Sphere')
        b = qt.Bloch(fig=fig, axes=ax)

        states = [qt.Qobj(state[:-1]) for state in self.states]
        # renormalize states:
        states = [state.unit() for state in states]
        psi0 = qt.basis(2, 0)
        # take the expectations of sigmax, sigmay, sigmaz on states:
        expects = [[qt.expect(qt.sigmax(), state), qt.expect(qt.sigmay(), state), qt.expect(qt.sigmaz(), state)] for state in states]
        expects = np.array(expects)
        b.add_points([expects[:,0], expects[:,1], expects[:,2]], meth='l')

        fig.tight_layout()
        b.save(f'figures/{self.pulse_name}-bloch.png')


    def plot(self, title='expects'):
        assert self.evolved, "System not yet evolved. Nothing to plot."
        plt.figure()

        states = [qt.Qobj(state[:-1]) for state in self.states]
        # renormalize states:
        states = [state.unit() for state in states]
        psi0 = qt.basis(2, 0)
        # take the expectations of sigmax, sigmay, sigmaz on states:
        expects = [[qt.expect(qt.sigmax(), state), qt.expect(qt.sigmay(), state), qt.expect(qt.sigmaz(), state)] for state in states]
        expects = np.array(expects)
        plt.plot(self.t_points, expects[:,0], label=r'$\langle \sigma_x \rangle$')
        plt.plot(self.t_points, expects[:,1], label=r'$\langle \sigma_y \rangle$')
        plt.plot(self.t_points, expects[:,2], label=r'$\langle \sigma_z \rangle$')
        plt.xlabel('Time')
        plt.ylabel('Expectation values')
        plt.title('Expectation values vs time')
        plt.legend()
        plt.savefig(f'figures/{self.pulse_name}-{title}.png')
