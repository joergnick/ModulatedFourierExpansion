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

K = 10
Nx = 50
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
Nt = 2**14
tauref = T*1.0/Nt
Am_rho = 2
Am_eps = 2
plt.figure()
for rhoind in range(Am_rho):
    for epsind in range(Am_eps):
        rho = 0.4*2**(-rhoind)
        #rho = 0.4*2**(-rhoind)
        eps = 0.4*10**(-epsind)
        filename = 'data/z_K_rho_'+str(rho)+'_eps_'+str(eps)+'.npy'
        if os.path.isfile(filename) and False:
            z_K = np.load(filename)
        else:
            print('######## NEW RUN, rho = '+str(rho)+', eps = '+str(eps)+' ###############')
            def eta(t):
               return 1+2*rho*np.cos(t/eps)
            start = time.time()
            refs = td_solver(f,eta,T,Nt,Nx,None,None,deg=50)
            mfe_vals,z_K = make_mfe_sol(rho,eps,Nt,T,Nx,K,f,-1,xx)
            print("Error: "+str(np.linalg.norm(refs-mfe_vals)))
        #### Plotting
        plt.semilogy(rho**(np.abs(np.arange(-K,K+1,1))),linestyle='dashed')
        z_K_norms = np.array([1.0/np.sqrt(Nt*Nx)*np.linalg.norm(z_K[k*Nx:(k+1)*Nx,:]) for k in range(2*K+1)])
        plt.semilogy()
        plt.semilogy((rho*eps)**(np.abs(np.arange(-K,K+1,1))),linestyle='dashed',label=r"\varepsilon = "+str(eps)+r"\rho = "+str(rho))
        np.save(filename,z_K_norms,allow_pickle=True)
plt.savefig('plots/z_Ks_rho_'+str(rho)+'_eps_'+str(eps)+'.pdf')
 
        #np.save('data/errs_rho_'+str(rho)+'_eps_'+str(eps)+'_smallNx.npy',res,allow_pickle=True)
      # Paper-ready plot style

