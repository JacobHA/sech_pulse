from matplotlib import pyplot as plt
import numpy as np
import qutip as qt

raising = qt.Qobj(np.array([[0, 1], [0, 0]]))
lowering = qt.Qobj(np.array([[0, 0], [1, 0]]))
psi_0 = qt.basis(2, 0)
psi_1 = qt.basis(2, 1)
omega_0 = 2*np.pi
omega = 2.*np.pi

unit_11 = qt.Qobj(np.array([[1,0,0],[0,0,0],[0,0,0]]))
unit_13 = qt.Qobj(np.array([[0,0,1],[0,0,0],[0,0,0]]))
unit_22 = qt.Qobj(np.array([[0,0,0],[0,1,0],[0,0,0]]))
unit_23 = qt.Qobj(np.array([[0,0,0],[0,0,1],[0,0,0]]))
unit_31 = qt.Qobj(np.array([[0,0,0],[0,0,0],[1,0,0]]))
unit_32 = qt.Qobj(np.array([[0,0,0],[0,0,0],[0,1,0]]))
unit_33 = qt.Qobj(np.array([[0,0,0],[0,0,0],[0,0,1]]))


def sech(t): 
    return 1/np.cosh(t)

def sech_amplitude(t, alpha, tau=1):
    return alpha/(np.pi * tau) * sech(t/tau)

def Bdot(t, beta, delta=0, tau=1):#Demokov-Kunikew
    # return beta*np.tanh(t/tau) / (np.pi * tau)
    return (beta + 2*delta + beta*np.tanh(t/tau)) / (np.pi * tau)

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

class TLS:
    def __init__(self, 
                 pulse_shape, 
                 psi_0=None, 
                 t_points=None, 
                 tau=1):
        self.pulse_name = pulse_shape
        self.psi_0 = psi_0 if psi_0 is not None else qt.basis(2, 0)
        self.t_points = t_points if t_points is not None else np.linspace(-10, 10, 100)
        self.t0 = self.t_points[0]
        self.tau = tau
        self.states = None
        self.expect = None
        self.final_fidelity = None
        self.evolved = False
        self.states = []
        self.expect_x = np.empty(0)
        self.expect_y = np.empty(0)
        self.expect_z = np.empty(0)
        self.be_expect_x = np.empty(0)
        self.be_expect_y = np.empty(0)
        self.be_expect_z = np.empty(0)
        self.total_time = np.empty(0)
        self.amplitudes = np.empty(0)
        self.detunings = np.empty(0)
    
    def evolve(self, 
               pulse_shape='sech',
               # density_matrix:
               initial_state= qt.thermal_dm(3, 0),
               phi=None,
               alpha=None,
               delta=0,
               thermal_temp=0, 
               global_phase=0,
               noise_level=0):
        self.phi = phi
        self.delta = delta
        self.beta = -2*delta
        if (phi is not None) and (alpha is None):
            self.alpha = alpha_func(self.phi, self.beta)
            self.phi = phi
        
        elif (phi is None) and (alpha is not None):
            self.alpha = alpha
            self.phi = np.sqrt(alpha**2 - self.beta**2)

        else:
            raise ValueError('Must specify exactly one of phi or alpha')

        if pulse_shape == 'square':
            self.amplitude = lambda t: square_amplitude(t, self.alpha, tau=self.tau)
            self.detuning = lambda t: square_detuning(t, self.beta, self.delta, tau=self.tau)
        elif pulse_shape == 'sech':
            self.amplitude = lambda t: sech_amplitude(t, self.alpha, tau=self.tau)
            self.detuning = lambda t: sech_detuning(t, beta=self.beta, delta=self.delta, tau=self.tau)
        elif pulse_shape == 'rabi':
            self.amplitude = lambda t: 1
            self.detuning = lambda t: (self.beta - self.delta)*t
        else:
            raise ValueError('pulse_shape must be "square" or "sech"')

        
        theta = np.pi/2
        xi = np.pi/2
        # param_0 = np.sin(theta/2) * np.exp(1j*xi)
        # param_1 = -np.cos(theta/2)
        param_0 = np.cos(theta/2)
        param_1 = np.sin(theta/2) * np.exp(1j*xi)

        const = np.exp(1j*global_phase) * 1/2
        H = [
            #  [unit_11, lambda t, args: -1/4 * (self.detuning(t))], 
            #  [unit_22, lambda t, args: -1/4 * (self.detuning(t))],
             [unit_13*const, lambda t, args: self.amplitude(t) * param_0 * np.exp(1j*self.detuning(t))], 
             [unit_31*const, lambda t, args: self.amplitude(t) * np.conj(param_0) * np.exp(-1j*self.detuning(t))],
             [unit_23*const, lambda t, args: self.amplitude(t) * param_1 * np.exp(1j*self.detuning(t))],
             [unit_32*const, lambda t, args: self.amplitude(t) * np.conj(param_1) * np.exp(-1j*self.detuning(t))],
            ]
        # H = [
        #     [unit_11, lambda t, args: 1/2 * (self.detuning(t))], 
        #     [unit_22, lambda t, args: -1/2 * (self.detuning(t))],
        #     [unit_13, lambda t, args: -1/2 * self.amplitude(t) * param_0 ], 
        #     [unit_31, lambda t, args: -1/2 * self.amplitude(t) * np.conj(param_0) ],
        #     [unit_23, lambda t, args: -1/2 * self.amplitude(t) * param_1 ],
        #     [unit_32, lambda t, args: -1/2 * self.amplitude(t) * np.conj(param_1) ],
        # ]
            
        result = qt.smesolve(H=H,
                            rho0=initial_state, #rho0
                            times=self.t_points,
                            # sc_ops=[noise_level*qt.sigmaz()],
                            # ntraj=1,
                            # e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()],
                            # method='homodyne',
                            options=qt.Options(store_states=True))
    
        # add times adjusted for start/stop:
        self.total_time = np.append(self.total_time, self.t_points + self.total_time[-1] - self.t_points[0] if len(self.total_time) > 0 else self.t_points)
        self.amplitudes = np.append(self.amplitudes, np.array([self.amplitude(t) for t in self.t_points]))
        self.detunings = np.append(self.detunings, np.array([self.detuning(t) for t in self.t_points]))
        self.states.append(result.states[0])
        # zip together the x, y, z components of the expectation values:
        #first project onto 0,1 subspace:
        # get the submatrixL

        self.expect_x = np.append(self.expect_x, np.array([qt.expect(qt.sigmax(), qt.Qobj(state[0:2,0:2]).unit()) for state in result.states[0]]))
        self.expect_y = np.append(self.expect_y, np.array([qt.expect(qt.sigmay(), qt.Qobj(state[0:2,0:2]).unit()) for state in result.states[0]]))
        self.expect_z = np.append(self.expect_z, np.array([qt.expect(qt.sigmaz(), qt.Qobj(state[0:2,0:2]).unit()) for state in result.states[0]]))
        # # 0, 1, E ---> D, B, E
        # rotate = qt.Qobj(np.array([[-param_1, param_0, 0],
        #                             [np.conj(param_0), np.conj(param_1), 0],
        #                             [0, 0, 1]]))
        rotate = qt.Qobj(np.array([[-param_1, np.conj(param_0), 0],
                                    [param_0, np.conj(param_1), 0],
                                    [0, 0, 1]]))

        # rotate = qt.Qobj(np.array([[np.conj(param_1), param_0, 0],
        #                             [-np.conj(param_0), param_1, 0],
        #                             [0, 0, 1]]))

        # make it unitary:
        rotate = rotate.unit()

        # now look in the bright excited subspace and similarly track the expectations:
        dressed_states = [rotate * state for state in result.states[0]]
        self.be_expect_x = np.append(self.be_expect_x, np.array([qt.expect(qt.sigmax(), qt.Qobj(state[1:3,1:3]).unit()) for state in dressed_states]))
        self.be_expect_y = np.append(self.be_expect_y, np.array([qt.expect(qt.sigmay(), qt.Qobj(state[1:3,1:3]).unit()) for state in dressed_states]))
        self.be_expect_z = np.append(self.be_expect_z, np.array([qt.expect(qt.sigmaz(), qt.Qobj(state[1:3,1:3]).unit()) for state in dressed_states]))

        self.final_fidelity = qt.fidelity(result.states[0][-1], initial_state)
        self.evolved = True
    
    def plot(self,title='dynamics'):
        assert self.evolved, "System not yet evolved. Nothing to plot."
        # loop over 0/1 anb b/e subspaces for expectation values:
        idx_to_subspace = {0: '01', 1: 'BE'}
        for idx, (x, y, z) in enumerate(zip([self.expect_x, self.be_expect_x], [self.expect_y, self.be_expect_y], [self.expect_z, self.be_expect_z])):
            # two subplots, one with expects, one with pulse
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            name = idx_to_subspace[idx]
            # increase all font size:
            plt.rcParams["font.size"] = 22
            ax1.plot(self.total_time, x, 'k', label=r'$\langle \sigma_x \rangle$', lw=3)
            ax1.plot(self.total_time, y, 'r', label=r'$\langle \sigma_y \rangle$', lw=3)
            ax1.plot(self.total_time, z, 'b', label=r'$\langle \sigma_z \rangle$', lw=3)
            ax1.legend(loc=0)
            # ax1.set_xlabel('Time', fontsize=22)
            # remove x ticks and place axes together:
            ax1.set_xticks([])

            ax1.set_ylabel('Expectation\nValues', fontsize=22)
            ax1.set_title('Sech Pulse Effect')
            # put some vspace between the two plots

            ax2.plot(self.total_time, [a*np.cos(detuning)
                                      for a, detuning in zip(self.amplitudes, self.detunings)],
                                        'g', label=r'$\Omega(t)$', lw=3)
            # add envelope:
            ax2.plot(self.total_time, self.amplitudes, 'g--', label=r'$\Omega(t)$', lw=2, alpha=0.7)
            ax2.set_xlabel('Time', fontsize=22)
            ax2.set_ylabel('Pulse amplitude', fontsize=22)
            # ax2.set_title('Pulse amplitude')
            # remove any vertical space between the two plots
            plt.subplots_adjust(hspace=0)
            fig.tight_layout()
            plt.savefig(f'figures/{self.pulse_name}-{title}-{name}.png')

    def plot_bloch(self, title='bloch-path'):
        assert self.evolved, "System not yet evolved. Nothing to plot."
        idx_to_subspace = {0: '01', 1: 'BE'}
        for idx, (x, y, z) in enumerate(zip([self.expect_x, self.be_expect_x], [self.expect_y, self.be_expect_y], [self.expect_z, self.be_expect_z])):
            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='3d'))
            name = idx_to_subspace[idx]

            ax.axis('square')
            # Plot on Bloch sphere
            b = qt.Bloch(fig=fig, axes=ax)
            b.add_points([x, y, z], meth='l')

            fig.tight_layout()
            b.save(f'figures/{self.pulse_name}-{name}-bloch.png')
            plt.close()