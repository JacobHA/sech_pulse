# apply sigmay to the initial state 
import qutip as qt
x0 = qt.basis(2, 0)

def ham(t, args):
    return 0.5*qt.sigmay()

results = qt.sesolve(ham, x0, [0, 1], args={})
print(results.states[-1])