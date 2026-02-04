import os.path
import numpy as np
import scipy
import sys
sys.path.append('./cqToolbox')
from numpy.polynomial.legendre import leggauss, legval, legder
import matplotlib.pyplot as plt
import matplotlib as mpl
from mfe_direct_helpers import make_mfe_sol
#from mfe_helpers_finite_differences import create_rhs,time_harmonic_solve,make_mfe_sol
from cqToolbox.linearcq import Conv_Operator

from mfe_ref_fd import td_solver

import time

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
    "text.usetex": True,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
})
K = 10
Nx = 1000
etas = np.zeros((2*K+1,))

#rho = 0.4
#eps = 0.0001

def f(x,t):
    t0 = 1
    #return np.exp(-10*x**2)*np.exp(-5*(t-t0)**2)
    a = 100
    b = 10
    return (np.exp(-a*(x-0.5)**2)*np.exp(-b*(t-t0)**2)-np.exp(-a*(x-0.5)**2)*np.exp(-b*(t-t0-0.1)**2))
    #return (np.exp(-a*(x-0.5)**2)*np.exp(-b*(t-t0)**2)-np.exp(-a*(x-0.5)**2)*np.exp(-b*(t-t0-0.1)**2))

def feps(x,t):
    return 2*np.cos(t/eps)*f(x,t)

xx = np.linspace(0,1,Nx)
T = 4
## Compute reference solution
#Ntref = 13*2**7
Nt = 2**11
tauref = T*1.0/Nt
Am_rho = 2
Am_eps = 2
fig, ax = plt.subplots(figsize=(6.0, 4.0))
mlist=[['o','d'],['p','s']]

for rhoind in range(Am_rho):
    for epsind in range(Am_eps):
        rho = np.round(0.4*10**(-rhoind),5)
        #rho = 0.4*2**(-rhoind)
        eps = np.round(0.4*10**(-epsind),5)
        filename = 'data/z_K_rho_'+str(rho)+'_eps_'+str(eps)+'.npy'
        if os.path.isfile(filename) and False:
            z_K_norms = np.load(filename)

        else:
            print('######## NEW RUN, rho = '+str(rho)+', eps = '+str(eps)+' ###############')
            def eta(t):
               return 1+2*rho*np.cos(t/eps)
            start = time.time()
            refs = td_solver(f,eta,T,Nt,Nx,None,None,deg=50)
            mfe_vals,z_K = make_mfe_sol(rho,eps,Nt,T,Nx,K,f,-1,xx)
            z_K_norms = np.array([1.0/np.sqrt(Nt*Nx)*np.linalg.norm(z_K[k*Nx:(k+1)*Nx,:]) for k in range(2*K+1)])
            print("Error: "+str(np.linalg.norm(refs-mfe_vals)))
        k = np.arange(0, K + 1)
        k_ref = np.arange(0, K + 1)
        #### Plotting
        ax.semilogy(
            k,
            z_K_norms[K:],
            marker=mlist[rhoind][epsind],
            linestyle='-',
            label=r"$\|z_k\|$, "+ rf"$\varepsilon={eps},\,\rho={rho}$"
        )
        ax.semilogy(
        k_ref,
        z_K_norms[K] * (T * rho * eps)**np.abs(k_ref),
        linestyle='--',
        marker=mlist[rhoind][epsind],
        color='black',
        label=(
        r"Reference: "
        r"$ (T \rho \varepsilon)^{k}$, "
        #rf"$\varepsilon={eps},\,\rho={rho}$" 
        ))
        # axis labels
        ax.set_xlabel(r"Index $k$")
        ax.set_ylabel(r"$L^2$-norm")
        
        # limits
        ax.set_ylim(1e-18, 1e-1)
        
        # legend
        ax.legend(loc="lower left", frameon=True)
        
        # layout
        fig.tight_layout()
       # plt.semilogy(z_K_norms[K:])
       # plt.semilogy(z_K_norms[K]*(T*rho*eps)**(np.abs(np.arange(0,K+1,1))),linestyle='dashed',label=r"\varepsilon = "+str(eps)+r"\rho = "+str(rho))
       # plt.ylim([1e-15,1e-1])
        np.save(filename,z_K_norms,allow_pickle=True)
fig.savefig('plots/z_Ks_decay.pdf')

        #np.save('data/errs_rho_'+str(rho)+'_eps_'+str(eps)+'_smallNx.npy',res,allow_pickle=True)
      # Paper-ready plot style

