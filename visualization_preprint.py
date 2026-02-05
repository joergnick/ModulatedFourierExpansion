import numpy as np
import scipy
import sys
sys.path.append('./cqToolbox')
from numpy.polynomial.legendre import leggauss, legval, legder
import matplotlib.pyplot as plt
import matplotlib as mpl
from mfe_direct_helpers import make_mfe_sol
#from mfe_helpers_finite_differences import create_rhs,time_harmonic_solve,make_mfe_sol

from mfe_ref_fd import td_solver

import time

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

K = 3
Nx = 500
etas = np.zeros((2*K+1,))

#rho = 0.4
#eps = 0.0001

def f(x,t):
    t0 = 1
    #return np.exp(-10*x**2)*np.exp(-5*(t-t0)**2)
    a = 100
    b = 10
    return (np.exp(-a*(x-0.5)**2)*np.exp(-b*(t-t0)**2)-np.exp(-a*(x-0.5)**2)*np.exp(-b*(t-t0-0.1)**2))

def feps(x,t):
    return 2*np.cos(t/eps)*f(x,t)

xx = np.linspace(0,1,Nx)
T = 5
## Compute reference solution
#Ntref = 13*2**7
rho = 0.4
eps = 0.04
print('######## NEW RUN, rho = '+str(rho)+', eps = '+str(eps)+' ###############')
#eps = 0.01*10**(-epsind)
def eta(t):
    return 1+2*rho*np.cos(t/eps)
start = time.time()
#refs = np.load('data/ref.npy')
#end   = time.time()
#print("Duration computation reference solution: ", end-start)
#refs,z_K = make_mfe_sol(rho,eps,Ntref,T,Nx,K,f,-1,-1)

Nt = 256
tau = T*1.0/Nt

mfe_vals,z_K = make_mfe_sol(rho,eps,Nt,T,Nx,K,f,-1,xx)

extent = [0, 4, 0, 1] 
plt.figure(0,figsize=(8,4))
plt.imshow(np.real(mfe_vals),aspect='auto',extent=extent)
plt.xlabel(r"$t$", fontsize=14)
plt.ylabel(r"$x$", fontsize=14)
plt.title(r"$u$", fontsize=16,pad=10)
#plt.title(r"$u$")
plt.colorbar()
plt.savefig('plots/visualization.pdf')
plt.close()

# First subplot
plt.figure(figsize=(8,12))
ax1 = plt.subplot(3, 1, 1)
im1 = ax1.imshow(np.abs(z_K[3*Nx:4*Nx,:]), aspect='auto', extent=extent)
ax1.set_xlabel(r"$t$", fontsize=14)
ax1.set_ylabel(r"$x$", fontsize=14)
ax1.set_title(r"$|z_0|$", fontsize=16,pad=10)
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.ax.tick_params(labelsize=12)

# Second subplot
ax2 = plt.subplot(3, 1, 2)
im2 = ax2.imshow(np.abs(z_K[4*Nx:5*Nx,:]), aspect='auto', extent=extent)
ax2.set_xlabel(r"$t$", fontsize=14)
ax2.set_ylabel(r"$x$", fontsize=14)
ax2.set_title(r"$|z_1|$", fontsize=16,pad=10)
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.ax.tick_params(labelsize=12)

# Third subplot
ax3 = plt.subplot(3, 1, 3)
im3 = ax3.imshow(np.abs(z_K[5*Nx:6*Nx,:]), aspect='auto', extent=extent)
ax3.set_xlabel(r"$t$", fontsize=14)
ax3.set_ylabel(r"$x$", fontsize=14)
ax3.set_title(r"$|z_2|$", fontsize=16,pad=10)
cbar3 = plt.colorbar(im3, ax=ax3)
cbar3.ax.tick_params(labelsize=12)

plt.tight_layout(pad=3.0) 
plt.savefig('plots/zKs.pdf')
plt.close()
