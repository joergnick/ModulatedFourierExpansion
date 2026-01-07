import numpy as np
import matplotlib.pyplot as plt
from finite_difference_cn_helpers import mass_matrix,stiffness_matrix,crank_nicolson_step_variable_c
# Parameters
L = 1.0
Nx = 200
Ne = Nx - 1
Nt = 501
T = 1
dt = T / Nt

x = np.linspace(0, L, Nx)
h = x[1] - x[0]

u0 = np.exp(-300*(x-0.5)**2)
v0 = np.zeros_like(u0)

# Example: wave speed varies in space and time
def c_func(x, t):
    return 1.0 + 0.1 * np.sin(2*np.pi*x) * np.cos(5*t)
    #return 1.0 + 0.5 * np.sin(2*np.pi*x) * np.cos(5*t)

plt.plot(x, u0, label="t=0")
u, v = crank_nicolson_step_variable_c( h, c_func, dt, Nt, u0, v0, x)
plt.plot(x, u, label=f"t={T}")
plt.legend()
plt.xlabel("x")
plt.ylabel("u")
plt.title("1D FEM Wave Equation - Crank–Nicolson, c(x,t)")
plt.savefig("test_fem_variable_c.png")
