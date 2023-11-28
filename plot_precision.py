# Plot the data at precision_vs_delta.npy
import numpy as np
import matplotlib.pyplot as plt

data = np.load('precision_vs_delta.npy')
deltas = data[0]
precision_list = data[1]
plt.figure()
# ignore infinities:
# precision_list = [precision for precision in precision_list if precision != np.inf]
# deltas = [deltas for precision in precision_list if precision != np.inf]
plt.plot(deltas, precision_list)
plt.xlabel('delta')
plt.ylabel(r'Width at 99\% fidelity')
plt.title('Error vs delta')
plt.savefig('error.png')
plt.savefig('precision_vs_delta.png')
plt.show()