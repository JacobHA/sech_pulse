import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def main():
    sns.set_style("darkgrid")
    # poster mode:
    plt.rcParams["figure.figsize"] = (12,8)
    plt.rcParams["font.size"] = 22
    x = np.linspace(-10,10,1000)
    sech = 1/np.cosh(x)
    gaussian = np.exp(-x**2/4)
    plt.figure()
    plt.title("Comparison of Hyperbolic Secant and Gaussian Pulses")
    plt.plot(x,sech,label=r"$\mathrm{sech}(t)=\frac{2}{e^t + e^{-t}}$", lw=4, color='k')
    plt.plot(x,gaussian,label=r"$\mathrm{Gaussian}(t)=e^{-t^2}$", lw=4, color='r')
    plt.xlabel(r"Pulse Time, $t$ [a.u.]")
    plt.ylabel(r"Pulse Amplitude, $\Omega(t)$ [a.u.]")
    plt.legend(fontsize=22)
    plt.savefig("../sech.png", dpi=300)
    plt.close()



if __name__ == "__main__":
    main()