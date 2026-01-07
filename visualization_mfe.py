import numpy as np
import scipy
import sys
sys.path.append('./cqToolbox')
from spectral_Galerkin import spectral_galerkin_matrices,project_legendre_from_values,precomp_project,project_legendre
from numpy.polynomial.legendre import leggauss, legval, legder
import matplotlib.pyplot as plt
import matplotlib as mpl
from mfe_helpers import create_rhs,time_harmonic_solve,make_mfe_sol
from cqToolbox.linearcq import Conv_Operator

from mfe_ref import td_solver


mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

K = 5
Nx = 30
etas = np.zeros((2*K+1,))

#rho = 0.4
#eps = 0.0001
def f(x,t):
    t0 = 1
    return np.exp(-20*x**2)*np.exp(-10*(t-t0)**2)

deg = 50
x_g,w   = leggauss(deg)

nxplot=200
xx = np.linspace(-1,1,nxplot)
T =5
## Compute reference solution
#Ntref = 13*2**7
Ntref = 2**13
tauref = T*1.0/Ntref
for rhoind in range(1):
    for epsind in range(1):
        rho = 0.45*2**(-rhoind)
        #rho = 0.4*2**(-rhoind)
        eps = 0.1*10**(-epsind)
        #eps = 0.01*10**(-epsind)
        def eta(t):
            return 1+2*rho*np.cos(t/eps)
        refs,sol,M,A = td_solver(f,eta,T,Ntref,Nx,None,None,deg=deg,return_sg=True,nx=nxplot)
        #CN_vals,sol,M,A = td_solver(f,eta,T,Nt,Nx,None,None,deg=deg,return_sg=True)
        plt.figure(figsize=(6,4))
        plt.imshow(
            np.real(refs),
            aspect='auto',
            extent=[0, T, -1, 1],   # x from 0→1, y from 0→1 (top flipped to usual orientation)
            cmap='viridis'
        )
        plt.colorbar(label='Value')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$x$')
        plt.tight_layout()
        plt.savefig('plots/ref_rho'+str(rho)+'_eps'+str(eps)+'.pdf')
        plt.close()

        plt.figure()
        ttref = np.linspace(0,T,Ntref+1) 
        tauref = T*1.0/Ntref
        energies_ref = np.array([sol[Nx:,j].T @ M @ sol[Nx:,j]+eta(tauref*j)*sol[:Nx,j].T @ A @ sol[:Nx,j] for j in range(len(refs[0,:]))])
        print('Reference solution plotted.')
        #mfe_refs = refs
        mfe_refs,z_K = make_mfe_sol(rho,eps,Ntref,T,Nx,K,f,deg,xx)
        Am_Nt = 2
        
        Nts  = np.zeros((Am_Nt,))
        cn_errs = np.zeros((Am_Nt,))
        mfe_errs = np.zeros((Am_Nt,))
        mfe_cn_diff = np.zeros((Am_Nt,))
        for j in range(Am_Nt):
            Nt = 2**11*2**j
            tau = T*1.0/Nt
            CN_vals,sol,M,A = td_solver(f,eta,T,Nt,Nx,None,None,deg=deg,return_sg=True,nx=nxplot)
            rhs = create_rhs(Nt,T,Nx,K,f,precomp = None,deg=deg)
            speed = int(Ntref/Nt)
        
            Nts[j]  = Nt
            #mfe_vals = CN_vals
            mfe_vals,z_K,mfe_sol = make_mfe_sol(rho,eps,Nt,T,Nx,K,f,deg,xx,return_sg=True)
            mfe_errs[j] = 1.0/np.sqrt(Nt*len(xx))*np.linalg.norm(mfe_refs[:,::speed]-CN_vals)
            #mfe_errs[j] = 1.0/np.sqrt(Nt*len(xx))*np.linalg.norm(mfe_refs[:,::speed]-mfe_vals)
            cn_errs[j] = 1.0/np.sqrt(Nt*len(xx))*np.linalg.norm(refs[:,::speed]-CN_vals)
            #print("Reference error: ",np.linalg.norm(mfe_refs-refs))
            print("MFE error: ",mfe_errs)
            print("CN error: ",cn_errs)
            mfe_cn_diff[j]= 1.0/np.sqrt(Nt*len(xx))*np.linalg.norm(mfe_vals-CN_vals)

            plt.subplot(3,1,1)
            plt.imshow(np.real(mfe_vals),aspect='auto')
            plt.clim([-1,3])
            plt.colorbar()
            plt.subplot(3,1,2)
            plt.imshow(np.real(CN_vals),aspect='auto')
            plt.clim([-1,3])
            plt.colorbar()
            plt.subplot(3,1,3)
            plt.imshow(np.real(CN_vals-mfe_vals),aspect='auto')
            plt.clim([-1,3])
            plt.colorbar()

            #energies_sol = np.array([sol[Nx:,j].T @ M @ sol[Nx:,j]+eta(tau*j)*sol[:Nx,j].T @ A @ sol[:Nx,j] for j in range(len(sol[0,:]))])
            #energies_sol = np.array([sol[Nx:,j].T @ M @ sol[Nx:,j]+eta(tau*j)*sol[:Nx,j].T @ A @ sol[:Nx,j] for j in range(len(sol[0,:]))])
            #dtmfe = 0*mfe_sol
            #for j in range(Nt-1):
            #    dtmfe[:,j]= (mfe_sol[:,j+1]-mfe_sol[:,j])/tau
            #dtmfe[:,-1] = dtmfe[:,-2]
            #energies_mfe = np.array([dtmfe[:,j].T @ M @ dtmfe[:,j]+eta(tau*j)*mfe_sol[:,j].T @ A @ mfe_sol[:,j] for j in range(len(sol[0,:]))])
            #tt = np.linspace(0,T,Nt+1) 
            #plt.semilogy(tt,energies_sol,label=r'TR $\tau = $'+str(tau))
            #plt.semilogy(tt,energies_mfe,label=r'MFE-TR $\tau = $'+str(tau))
            #plt.semilogy(np.linalg.norm(mfe_vals-CN_vals,axis = 0))
            #plt.semilogy(10**(-6)*np.exp(1.0/np.pi*rho/eps*tt),linestyle='dashed')
            #plt.imshow(np.log(np.abs(mfe_vals-CN_vals)),aspect='auto')
            #plt.colorbar()
        
       # plt.semilogy(ttref,energies_ref,label='Reference',linestyle='dashed')
       # plt.ylim([10**(-3),10**(1)])
       # plt.legend()
        plt.savefig('plots/visualization_rho'+str(rho)+'_eps'+str(eps)+'.pdf')
        plt.close()

 
        #plt.figure() 
        #plt.semilogy(rho**(np.abs(np.arange(-K,K+1,1))),linestyle='dashed')

        #plt.semilogy(np.array([1.0/np.sqrt(Nt*Nx)*np.linalg.norm(z_K[k*Nx:(k+1)*Nx,:]) for k in range(2*K+1)]))
        #plt.semilogy((rho*eps)**(np.abs(np.arange(-K,K+1,1))),linestyle='dashed')
        #plt.savefig('z_Ks.pdf')
        #plt.close()



      # Paper-ready plot style

      # --- Matlab-like styling ---
        mpl.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 12,
            'axes.linewidth': 1.2,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'lines.linewidth': 1.6,
            'lines.markersize': 6,
            'grid.linestyle': '--',
            'grid.linewidth': 0.6,
        })
        
        # Matlab default color cycle
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
            '#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F'
        ])
        
        plt.figure(figsize=(6, 4), dpi=300)
        
        # Main curves
        plt.loglog(T / Nts, mfe_errs, '-o', label='MFE-TR')
        plt.loglog(T / Nts, cn_errs, '-s', label='TR')
        
        # Reference slopes
        #plt.loglog(T / Nts, T / Nts, '--', label=r'$h$')
        plt.loglog(T / Nts, (T / Nts)**2, '--', label=r'$\tau^2$')
        
        # Labels and axis limits
        plt.title(r'Time convergence')
        plt.xlabel(r'Time step size $\tau$')
        plt.ylabel(r'$L^2-$Error')
        #plt.ylim(1e-5, 2e-1)
        
        # Grid + legend
        plt.grid(True, which='both')
        plt.legend()
        
        plt.tight_layout()
       # plt.figure(figsize=(6, 4), dpi=300)
       # 
       # plt.loglog(T / Nts, mfe_errs, label='MFE', linewidth=2)
       # plt.loglog(T / Nts, cn_errs, label='CN', linewidth=2)
       # 
       # # Reference slopes
       # #plt.loglog(T / Nts, T / Nts, '--', label=r'$\tau$', linewidth=1.5)
       # plt.loglog(T / Nts, (T / Nts)**2, '--', label=r'$\tau^2$', linewidth=1.5)
       # 
       # # Labels, legend, formatting
       # plt.xlabel('Time step size $h$', fontsize=12)
       # plt.ylabel('Error', fontsize=12)
       # plt.title('Convergence of MFE and CN Schemes', fontsize=13)
       # 
       # plt.grid(True, which='both', ls=':', alpha=0.7)
       # plt.legend(fontsize=11)
       # 
       # plt.tight_layout()
       # plt.figure()
       # plt.loglog(T*1.0/Nts,mfe_errs,label='MFE')
       # plt.loglog(T*1.0/Nts,cn_errs,label='CN')
       # #plt.loglog(T*1.0/Nts,mfe_cn_diff,label='CN-MFE')
       # plt.loglog(T*1.0/Nts,(T*1.0/Nts),label='h',linestyle='dashed')
       # plt.loglog(T*1.0/Nts,(T*1.0/Nts)**2,label='h^2',linestyle='dashed')
       # plt.legend()
        plt.savefig('plots/td_convergence_rho'+str(rho)+'_eps'+str(eps)+'.pdf')
        plt.close()
        
       #breakpoint()
        #vals = np.array([f(xx,tau*j) for j in range(Nt+1)]).T