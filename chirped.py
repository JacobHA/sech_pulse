import matplotlib.pyplot as plt
from utils import sech_amplitude, sech_detuning
import numpy as np

t_points = np.linspace(-15, 15, 10_000)

plt.figure()
detune = np.real(sech_detuning(t_points, 10, 1))
envelope = sech_amplitude(t_points, 2*np.pi)
plt.plot(t_points, envelope * np.cos(detune))
plt.plot(t_points, envelope)
plt.show()
