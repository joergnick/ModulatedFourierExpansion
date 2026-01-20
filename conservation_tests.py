import numpy as np
import scipy
import sys
sys.path.append('./cqToolbox')
from numpy.polynomial.legendre import leggauss, legval, legder
import matplotlib.pyplot as plt
import matplotlib as mpl
#from mfe_helpers_finite_differences import create_rhs,time_harmonic_solve,make_mfe_sol
from mfe_direct_helpers import make_mfe_sol as make_mfe_sol_direct
from mfe_direct_backup_helpers import make_mfe_sol
#from mfe_helpers_finite_differences import make_mfe_sol
from cqToolbox.linearcq import Conv_Operator

from mfe_ref_fd import td_solver

import time

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

K = 10
Nx = 100
etas = np.zeros((2*K+1,))

#rho = 0.4
#eps = 0.0001
xx = np.linspace(0,1,Nx)
T = 4
## Compute reference solution
#Ntref = 13*2**7
Ntref = 128
ttref = np.linspace(0,T,Ntref+1)
tauref = T*1.0/Ntref
Am_rho = 1
Am_eps = 1
perf = np.zeros((Am_rho,Am_eps))
Am_Nt = 8
taus = np.zeros((Am_Nt,))
rho = 0.4
#rho = 0.4*2**(-rhoind)
eps = 0.02

def f(x,t):
    t0 = 1
    #return np.exp(-10*x**2)*np.exp(-5*(t-t0)**2)
    a = 100
    b = 10
    return (np.exp(-a*(x-0.5)**2)*np.exp(-b*(t-t0)**2)-np.exp(-a*(x-0.5)**2)*np.exp(-b*(t-t0-0.1)**2))

def feps(x,t):
    return 2*np.cos(t/eps)*f(x,t)


print('######## NEW RUN, rho = '+str(rho)+', eps = '+str(eps)+' ###############')
#eps = 0.01*10**(-epsind)
def eta(t):
    return 1+2*rho*np.cos(t/eps)

start = time.time()
#refs = td_solver(f,eta,T,Ntref,Nx,None,None,deg=50)
#np.save('data/ref.npy',refs)
#refs = np.load('data/ref.npy')
#end   = time.time()
#print("Duration computation reference solution: ", end-start)
#print("Computed reference solution.")
#Kref = 10
#refs,z_K = make_mfe_sol(rho,eps,Ntref,T,Nx,K,f,-1,-1)

Nts  = np.zeros((Am_Nt,))
cn_errs = np.zeros((Am_Nt,))
mfe_errs = np.zeros((Am_Nt,))
mfe_cn_diff = np.zeros((Am_Nt,))
res = {}

plt.figure()

for j in range(Am_Nt):
    Nt = 64*2**j
    tt = np.linspace(0,T,Nt+1)
    print("Computation: "+str(j+1)+" of "+str(Am_Nt)+" N = "+str(Nt))
    tau = T*1.0/Nt
    taus[j] = tau
    start = time.time()
    mfe_dir,z_K = make_mfe_sol_direct(rho,eps,Nt,T,Nx,K,f,-1,xx)
    end = time.time()
    print("Duration MFE-TR: "+str(end-start))
    start = time.time()
    CN_vals = td_solver(feps,eta,T,Nt,Nx,None,None,deg=-1)

    end = time.time()
    print("Duration TR: "+str(end-start))
    speed = int(Ntref/Nt)
    Nts[j]  = Nt
    #mfe_vals = mfe_dir
    #z_K = z_K_dir
    mfe_vals,z_K_ = make_mfe_sol(rho,eps,Nt,T,Nx,K,f,-1,xx)
   # mfe_errs[j] = 1.0/np.sqrt(Nt*len(xx))*np.linalg.norm(refs[:,::speed]-mfe_vals)
   #
   #  cn_errs[j] = 1.0/np.sqrt(Nt*len(xx))*np.linalg.norm(refs[:,::speed]-CN_vals)
   # mfe_cn_diff[j]= 1.0/np.sqrt(Nt*len(xx))*np.linalg.norm(mfe_vals-CN_vals)
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.semilogy(tt,np.linalg.norm(mfe_vals,axis = 0),label="MFE, N = "+ str(Nt))
    plt.ylim([1e-6,1e2])
    plt.subplot(2,2,2)
    plt.semilogy(tt,np.linalg.norm(mfe_vals-CN_vals,axis = 0),label="MFE-CQ, N = "+ str(Nt))
    #plt.legend()
    plt.ylim([1e-10,1e-2])
    plt.subplot(2,2,3)
    plt.semilogy(tt,np.linalg.norm(mfe_dir,axis = 0),label="MFE, N = "+ str(Nt))
    plt.ylim([1e-6,1e2])
    plt.subplot(2,2,4)
    plt.semilogy(tt,np.linalg.norm(mfe_dir-CN_vals,axis = 0)/(np.linalg.norm(CN_vals,axis = 0)+1),label="MFE-Dir, N = "+ str(Nt))
    #plt.legend()
    plt.ylim([1e-10,1e0])
    #print("Difference: ",np.linalg.norm(z_K_dir-z_K))
    #plt.plot(ttref,np.linalg.norm(refs,axis = 0),linestyle='dashed')
    #plt.ylim([0,0.5])
   # print('MFE errs: ',mfe_errs)
   # print('CN errs: ',cn_errs)
    plt.savefig('time_plot_mfe.pdf')
    plt.figure(5)
    plt.semilogy(tt,np.linalg.norm(mfe_dir,axis = 0),label="CN, N = "+ str(Nt))
    plt.semilogy(tt,np.linalg.norm(CN_vals,axis = 0),label="CN, N = "+ str(Nt),linestyle='dashed')
    plt.savefig('sol.pdf')
    plt.figure(2,figsize=(20,14))
    am_K_plot = 16
    for k_ind in range(am_K_plot):
        plt.subplot(4,4,k_ind+1)
        plt.title("z_"+str(k_ind))
      #  err_z = z_K_dir[:(2*K+1)*Nx,:]-z_K
        plt.semilogy(tt,np.linalg.norm(z_K[(k_ind+K)*Nx:(k_ind+K+1)*Nx,:],axis = 0))
        plt.ylim([1e-16,1e2])
    plt.savefig('z_K.pdf')
    print(np.array([1.0/np.sqrt(Nt*Nx)*np.linalg.norm(z_K[k*Nx:(k+1)*Nx,:]) for k in range(2*K+1)]))
    #print(np.array([1.0/np.sqrt(Nt*Nx)*np.linalg.norm(z_K_dir[k*Nx:(k+1)*Nx,:]) for k in range(2*K+1)]))
 

#plt.plot(ttref,np.linalg.norm(refs,axis = 0),linestyle='dashed',label="MFE-TR, Nref = "+str(Ntref))
