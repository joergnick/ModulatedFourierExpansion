import numpy as np
import scipy
import sys
sys.path.append('./cqToolbox')
from spectral_Galerkin import spectral_galerkin_matrices,project_legendre_from_values,precomp_project,project_legendre
from numpy.polynomial.legendre import leggauss, legval, legder
import matplotlib.pyplot as plt
from mfe_helpers import create_rhs,time_harmonic_solve,make_mfe_sol
from cqToolbox.linearcq import Conv_Operator

from mfe_ref import td_solver

K = 3
Nx = 50
etas = np.zeros((2*K+1,))
rho = 0.1
eps = 0.1
def f(x,t):
    t0 = 2
    return np.exp(-20*x**2)*np.exp(-10*(t-t0)**2)


deg = 50
x_g,w   = leggauss(deg)

nxplot=100
xx = np.linspace(-1,1,nxplot)
Nt = 2000
T = 10
tau = T*1.0/Nt
from mfe_helpers import make_mfe_sol
mfe_vals,z_K = make_mfe_sol(rho,eps,Nt,T,Nx,K,f,deg,xx)
#
#rhs = create_rhs(Nt,T,Nx,K,f,precomp = None,deg=deg)
##vals = np.array([f(xx,tau*j) for j in range(Nt+1)]).T
##vals = np.array([legval(xx,rhs[(K-1)*Nx:K*Nx,j]) for j in range(Nt+1)]).T
#precomp_sg = spectral_galerkin_matrices(Nx,deg=deg)
#def th_sys(s,b):
#    x_hat = time_harmonic_solve(s,b,K,Nx,rho,eps,precomp=precomp_sg)
#    return x_hat
#td_sys = Conv_Operator(th_sys,order=-1)
#
## CAREFUL OF INITIAL VALUES
#z_K = td_sys.apply_convol_no_symmetry(rhs,T,show_progress = True,cutoff = 10**(-7))
#mfe_sol = 1j*np.zeros((Nx,Nt+1))
#for j in range(Nt+1):
#    for k_ind in range(2*K+1):
#        k = k_ind-K
#        mfe_sol[:,j]  += np.exp(1j*k*tau*j/eps)*z_K[k_ind*Nx:(k_ind+1)*Nx, j]
#
#
#mfe_vals = np.array([legval(xx,mfe_sol[:,j]) for j in range(Nt+1)]).T

def eta(t):
    return 1+2*rho*np.cos(t/eps)
factor = 10
Ntref =Nt*factor

CN_vals = td_solver(f,eta,T,Ntref,Nx,None,None,deg=50)


print(np.linalg.norm(np.imag(mfe_vals)))

#vals = np.array([legval(xx,rhs[(K)*Nx:(K+1)*Nx,j]) for j in range(Nt+1)]).T
plt.subplot(3,1,1)
plt.imshow(np.real(mfe_vals))
plt.colorbar()
plt.subplot(3,1,2)
plt.imshow(np.real(CN_vals[:,::factor]))
plt.colorbar()
plt.subplot(3,1,3)
plt.imshow(np.real(mfe_vals-CN_vals[:,::factor]))
plt.colorbar()
plt.savefig('approxs.pdf')
#f_hat = np.zeros(((2*K+1)*Nx,))
#x_hat = time_harmonic_solve(s,f_hat,K,Nx,rho,eps,precomp=None)
