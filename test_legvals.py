import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss, legval

# Function to project
f = lambda x: np.exp(x)

# Degree of projection
N = 4

# Gauss–Legendre quadrature points and weights
xg, w = leggauss(100)

# Compute coefficients: c_n = (2n+1)/2 ∫ f(x) P_n(x) dx
coeffs = np.array([(2*n+1)/2 * np.sum(w * f(xg) * legval(xg, [0]*n + [1])) for n in range(N+1)])

# Points for plotting
x_plot = np.linspace(-1, 1, 400)
f_exact = f(x_plot)
f_proj = legval(x_plot, coeffs)

# Plot
plt.figure(figsize=(7,4))
plt.plot(x_plot, f_exact, 'k-', lw=2, label='Exact $f(x)=e^x$')
plt.plot(x_plot, f_proj, 'r--', lw=2, label=f'Legendre Projection (N={N})')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.title('Projection of $f(x)=e^x$ onto Legendre Polynomial Space')
plt.savefig('rhs.pdf')