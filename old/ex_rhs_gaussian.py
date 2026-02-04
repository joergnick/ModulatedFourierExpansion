import numpy as np
import matplotlib.pyplot as plt

# parameters
T = 5.0
t0 = 2.0
Nx = 200
Nt = 200
Nmodes = 50

x = np.linspace(0, 1, Nx)
t = np.linspace(0, T, Nt)

# forcing components
def fx(x):
    return np.exp(-100*(x-0.5)**2)

def ft(t):
    return np.exp(-200*(t-t0)**2)

# precompute spatial coefficients
fn = np.zeros(Nmodes)
for n in range(1, Nmodes+1):
    fn[n-1] = 2*np.trapz(fx(x)*np.sin(n*np.pi*x), x)

# compute solution
U = np.zeros((Nt, Nx))

for i, ti in enumerate(t):
    for n in range(1, Nmodes+1):
        omega = n*np.pi
        s = np.linspace(0, ti, 300)
        qn = np.trapz(
            np.sin(omega*(ti - s))/omega * fn[n-1] * ft(s),
            s
        )
        U[i] += qn * np.sin(n*np.pi*x)

# plot solution as a space-time heatmap
plt.figure(figsize=(8,4))
plt.imshow(U, extent=[0,1,T,0], aspect='auto')
plt.xlabel("x")
plt.ylabel("t")
plt.title("Solution u(x,t)")
plt.colorbar(label="u")
plt.savefig('ex_sol.png')
