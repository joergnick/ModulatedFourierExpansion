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

K = 2
Nx = 30
etas = np.zeros((2*K+1,))

#rho = 0.4
#eps = 0.0001
def f(x,t):
    t0 = 2
    return np.exp(-10*x**2)*np.exp(-5*(t-t0)**2)

deg = 50
x_g,w   = leggauss(deg)

nxplot=100
xx = np.linspace(-1,1,nxplot)
T = 10
## Compute reference solution
Ntref = 2**12
tauref = T*1.0/Ntref
for rhoind in range(5):
    for epsind in range(5):
        rho = 0.4*2**(-rhoind)
        eps = 0.01*10**(-epsind)
        def eta(t):
            return 1+2*rho*np.cos(t/eps)
        refs = td_solver(f,eta,T,Ntref,Nx,None,None,deg=50)
        
        Am_Nt = 8
        
        Nts  = np.zeros((Am_Nt,))
        cn_errs = np.zeros((Am_Nt,))
        mfe_errs = np.zeros((Am_Nt,))
        mfe_cn_diff = np.zeros((Am_Nt,))
        for j in range(Am_Nt):
            Nt = 8*2**j
            CN_vals = td_solver(f,eta,T,Nt,Nx,None,None,deg=deg)
            rhs = create_rhs(Nt,T,Nx,K,f,precomp = None,deg=deg)
            speed = int(Ntref/Nt)
        
            Nts[j]  = Nt
            mfe_vals,z_K = make_mfe_sol(rho,eps,Nt,T,Nx,K,f,deg,xx)
            plt.semilogy(np.array([1.0/np.sqrt(Nt*Nx)*np.linalg.norm(z_K[k*Nx:(k+1)*Nx,:]) for k in range(2*K+1)]))
            mfe_errs[j] = 1.0/np.sqrt(Nt*len(xx))*np.linalg.norm(refs[:,::speed]-mfe_vals)
            cn_errs[j] = 1.0/np.sqrt(Nt*len(xx))*np.linalg.norm(refs[:,::speed]-CN_vals)
            mfe_cn_diff[j]= 1.0/np.sqrt(Nt*len(xx))*np.linalg.norm(mfe_vals-CN_vals)
        plt.figure() 
        plt.semilogy(rho**(np.abs(np.arange(-K,K+1,1))),linestyle='dashed')
        plt.semilogy((rho*eps)**(np.abs(np.arange(-K,K+1,1))),linestyle='dashed')
        plt.savefig('z_Ks.pdf')
        plt.close()
        
        plt.figure()
        plt.loglog(T*1.0/Nts,mfe_errs,label='MFE')
        plt.loglog(T*1.0/Nts,cn_errs,label='CN')
        #plt.loglog(T*1.0/Nts,mfe_cn_diff,label='CN-MFE')
        plt.loglog(T*1.0/Nts,(T*1.0/Nts),label='h',linestyle='dashed')
        plt.loglog(T*1.0/Nts,(T*1.0/Nts)**2,label='h^2',linestyle='dashed')
        plt.legend()
        plt.savefig('plots/errors_rho'+str(rho)+'_eps'+str(eps)+'.pdf')
        plt.close()
        #breakpoint()
        #vals = np.array([f(xx,tau*j) for j in range(Nt+1)]).T