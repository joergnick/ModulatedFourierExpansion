import numpy as np
import scipy
import sys
sys.path.append('./cqToolbox')
from numpy.polynomial.legendre import leggauss, legval, legder
import matplotlib.pyplot as plt
import matplotlib as mpl
from mfe_direct_helpers import make_mfe_sol as mfe1
from mfe_direct_backup_helpers import make_mfe_sol as mfe2
from mfe_helpers_finite_differences import make_mfe_sol as mfe3
from cqToolbox.linearcq import Conv_Operator

from mfe_ref_fd import td_solver

import time

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

K = 3
Nx = 50
etas = np.zeros((2*K+1,))

#rho = 0.4
#eps = 0.0001

def f(x,t):
    t0 = 1
    #return np.exp(-10*x**2)*np.exp(-5*(t-t0)**2)
    a = 100
    b = 10
    return 10**3*(np.exp(-a*(x-0.5)**2)*np.exp(-b*(t-t0)**2)-np.exp(-a*(x-0.5)**2)*np.exp(-b*(t-t0-0.1)**2))

def feps(x,t):
    return 2*np.cos(t/eps)*f(x,t)

xx = np.linspace(0,1,Nx)
T = 3
## Compute reference solution
#Ntref = 13*2**7
Ntref = 2**14
tauref = T*1.0/Ntref
Am_rho = 1
Am_eps = 1
perf = np.zeros((Am_rho,Am_eps))
Am_Nt = 9
taus = np.zeros((Am_Nt,))
for rhoind in range(Am_rho):
    for epsind in range(Am_eps):
        rho = 0.1*2**(-rhoind)
        #rho = 0.4*2**(-rhoind)
        eps = 0.01*10**(-epsind)
        print('######## NEW RUN, rho = '+str(rho)+', eps = '+str(eps)+' ###############')
        #eps = 0.01*10**(-epsind)
        def eta(t):
            return 1+2*rho*np.cos(t/eps)
        start = time.time()
        #np.save('data/ref.npy',refs)
        #refs = np.load('data/ref.npy')
        #end   = time.time()
        #print("Duration computation reference solution: ", end-start)
        print("Computed reference solution.")
        Kref = 10
        
        Nts  = np.zeros((Am_Nt,))
        cn_errs = np.zeros((Am_Nt,))
        mfe_errs = np.zeros((Am_Nt,))
        mfe_cn_diff = np.zeros((Am_Nt,))
        res = {}
        for j in range(Am_Nt):
            Nt = 4*2**j
            print("Computation: "+str(j+1)+" of "+str(Am_Nt)+" N = "+str(Nt))
            tau = T*1.0/Nt
            taus[j] = tau
            start = time.time()
            CN_vals = td_solver(f,eta,T,Nt,Nx,None,None,deg=-1)
            end = time.time()
            print("Duration TR: "+str(end-start))

            start = time.time()
            speed = int(Ntref/Nt)
            Nts[j]  = Nt
            mfe1_vals,z_K_1 = mfe1(rho,eps,Nt,T,Nx,K,f,-1,xx)
            mfe2_vals,z_K_2 = mfe2(rho,eps,Nt,T,Nx,K,f,-1,xx)
            mfe3_vals,z_K_3 = mfe3(rho,eps,Nt,T,Nx,K,f,-1,xx)
            end = time.time()
            print("||mfe1-mfe2|| = "+str(np.linalg.norm(mfe1_vals-mfe2_vals)))
            print("||mfe2-mfe3|| = "+str(np.linalg.norm(mfe2_vals-mfe3_vals)))
            print("||mfe1-mfe3|| = "+str(np.linalg.norm(mfe1_vals-mfe3_vals)))
            print("Duration MFE-TR: "+str(end-start))



