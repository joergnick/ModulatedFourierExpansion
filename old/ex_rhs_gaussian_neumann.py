import numpy as np
import matplotlib.pyplot as plt

# parameters
T = 5.0
t0 = 2.0
Nx = 200
Nt = 200
Nmodes = 50   # number of nonzero cosine modes

x = np.linspace(0, 1, Nx)
t = np.linspace(0, T, Nt)

# forcing components
def fx(x):
    return np.exp(-100*(x-0.5)**2)

def ft(t):
    return np.exp(-200*(t-t0)**2)

# spatial coefficients
f0 = np.trapz(fx(x), x)          # zero mode
fn = np.zeros(Nmodes)            # cosine modes n >= 1

for n in range(1, Nmodes+1):
    fn[n-1] = 2*np.trapz(fx(x)*np.cos(n*np.pi*x), x)

# solution array
U = np.zeros((Nt, Nx))

for i, ti in enumerate(t):
    s = np.linspace(0, ti, 300)

    # zero mode contribution
    q0 = np.trapz((ti - s) * f0 * ft(s), s)
    U[i] += q0

    # higher cosine modes
    for n in range(1, Nmodes+1):
        omega = n*np.pi
        qn = np.trapz(
            np.sin(omega*(ti - s))/omega * fn[n-1] * ft(s),
            s
        )
        U[i] += qn * np.cos(n*np.pi*x)

# plot space–time solution
plt.figure(figsize=(8,4))
plt.imshow(U, extent=[0,1,T,0], aspect='auto')
plt.xlabel("x")
plt.ylabel("t")
plt.title("Solution u(x,t) with Neumann BCs")
plt.colorbar(label="u")
plt.savefig('ex_neumann.png')
