import numpy as np
import scipy
import sys
sys.path.append('./cqToolbox')
#from spectral_Galerkin import spectral_galerkin_matrices,project_legendre_from_values,precomp_project,project_legendre
import matplotlib.pyplot as plt
from finite_difference_cn_helpers import finite_difference_matrices
from mfe_helpers_finite_differences import create_rhs,time_harmonic_solve,make_mfe_sol
from cqToolbox.linearcq import Conv_Operator

from mfe_ref_fd import td_solver

K = 4
Nx = 50
etas = np.zeros((2*K+1,))
rho = 0.4
eps = 0.1

def eta(t):
    return 1+2*rho*np.cos(t/eps)

def f(x,t):
    t0 = 2
    #return ((160000*(t - t0)**2 - 400)- eta(t)*(40000*(x - 0.5)**2 - 200)) * np.exp(-100*(x - 0.5)**2) * np.exp(-200*(t - t0)**2)
    #return np.exp(-100*(x-0.5)**2)*np.exp(-1*(t-t0)**2)
    return np.exp(-100*(x-0.5)**2)*np.exp(-200*(t-t0)**2)-np.exp(-100*(x-0.5)**2)*np.exp(-200*(t-t0-0.1)**2)
    #return np.exp(-100*(x-0.5)**2)*np.exp(-200*(t-t0)**2)
def u_ex(x,t):
    t0 = 2
    return np.exp(-100*(x-0.5)**2)*np.exp(-200*(t-t0)**2)

Nt = 700
T = 10
tau = T*1.0/Nt
mfe_vals,z_K = make_mfe_sol(rho,eps,Nt,T,Nx,K,f,None,None)

factor = 10
Ntref  = Nt*factor

CN_vals = td_solver(f,eta,T,Ntref,Nx,None,None,deg=50)

print(np.linalg.norm(np.imag(mfe_vals)))

#vals = np.array([legval(xx,rhs[(K)*Nx:(K+1)*Nx,j]) for j in range(Nt+1)]).T
xx = np.linspace(0,1,Nx)

A,M,Bl,Br = finite_difference_matrices(Nx)
bs = [u_ex(xx,tau*j) for j in range(Nt+1)]
plt.subplot(3,1,1)
plt.imshow(np.real(mfe_vals),aspect='auto')
plt.colorbar()
plt.subplot(3,1,2)
plt.imshow(np.real(CN_vals[:,::factor]),aspect='auto')
plt.colorbar()
plt.subplot(3,1,3)
plt.imshow(np.real(mfe_vals)-CN_vals[:,::factor],aspect='auto')
plt.colorbar()
plt.savefig('approxs.pdf')
#f_hat = np.zeros(((2*K+1)*Nx,))
#x_hat = time_harmonic_solve(s,f_hat,K,Nx,rho,eps,precomp=None)
