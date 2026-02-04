import numpy as np
import scipy

from spectral_Galerkin import spectral_galerkin_matrices,project_legendre_from_values,precomp_project,project_legendre
from numpy.polynomial.legendre import leggauss, legval, legder
import matplotlib.pyplot as plt

def f(x):
    return np.exp(-20*x**2)

deg = 100
x_g,w   = leggauss(deg)
Nx = 60
precomp = precomp_project(Nx,deg=deg)
#fcoeffs = project_legendre(f,Nx,deg=deg)
fcoeffs = project_legendre_from_values(f(x_g),Nx,precomp=precomp,deg=deg,x=x_g)
x_g = np.linspace(-1,1,100)
f_ex = f(x_g)
plt.plot(f_ex,linestyle='dashed')
plt.plot(legval(x_g,fcoeffs))
print(np.linalg.norm(f_ex-legval(x_g,fcoeffs)))
plt.savefig('rhs.pdf')

#f_hat = np.zeros(((2*K+1)*Nx,))
#x_hat = time_harmonic_solve(s,f_hat,K,Nx,rho,eps,precomp=None)
