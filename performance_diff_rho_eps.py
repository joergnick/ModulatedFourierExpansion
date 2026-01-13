import numpy as np
import scipy
import sys
sys.path.append('./cqToolbox')
from numpy.polynomial.legendre import leggauss, legval, legder
import matplotlib.pyplot as plt
import matplotlib as mpl
from mfe_helpers_finite_differences import create_rhs,time_harmonic_solve,make_mfe_sol
from cqToolbox.linearcq import Conv_Operator

from mfe_ref_fd import td_solver


mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

K = 3
Nx = 1000
etas = np.zeros((2*K+1,))

#rho = 0.4
#eps = 0.0001
def f(x,t):
    t0 = 2
    #return np.exp(-10*x**2)*np.exp(-5*(t-t0)**2)
    a = 100
    b = 10
    return np.exp(-a*(x-0.5)**2)*np.exp(-b*(t-t0)**2)-np.exp(-a*(x-0.5)**2)*np.exp(-b*(t-t0-0.1)**2)

xx = np.linspace(0,1,Nx)
T = 5
## Compute reference solution
#Ntref = 13*2**7
Ntref = 2**12
tauref = T*1.0/Ntref
Am_rho = 1
Am_eps = 1
perf = np.zeros((Am_rho,Am_eps))
Am_Nt = 9
taus = np.zeros((Am_Nt,))
for rhoind in range(Am_rho):
    for epsind in range(Am_eps):
        rho = 0.4*2**(-rhoind)
        #rho = 0.4*2**(-rhoind)
        eps = 0.001*10**(-epsind)
        print('######## NEW RUN, rho = '+str(rho)+', eps = '+str(eps)+' ###############')
        #eps = 0.01*10**(-epsind)
        def eta(t):
            return 1+2*rho*np.cos(t/eps)
        refs = td_solver(f,eta,T,Ntref,Nx,None,None,deg=50)
        #refs,z_K = make_mfe_sol(rho,eps,Ntref,T,Nx,K,f,-1,-1)
        
        Nts  = np.zeros((Am_Nt,))
        cn_errs = np.zeros((Am_Nt,))
        mfe_errs = np.zeros((Am_Nt,))
        mfe_cn_diff = np.zeros((Am_Nt,))
        res = {}
        for j in range(Am_Nt):
            Nt = 8*2**j
            tau = T*1.0/Nt
            taus[j] = tau
            CN_vals = td_solver(f,eta,T,Nt,Nx,None,None,deg=-1)
            rhs = create_rhs(Nt,T,Nx,K,f,precomp = None,deg=-1)
            speed = int(Ntref/Nt)
        
            Nts[j]  = Nt
            mfe_vals,z_K = make_mfe_sol(rho,eps,Nt,T,Nx,K,f,-1,xx)
            mfe_errs[j] = 1.0/np.sqrt(Nt*len(xx))*np.linalg.norm(refs[:,::speed]-mfe_vals)
            cn_errs[j] = 1.0/np.sqrt(Nt*len(xx))*np.linalg.norm(refs[:,::speed]-CN_vals)

            mfe_cn_diff[j]= 1.0/np.sqrt(Nt*len(xx))*np.linalg.norm(mfe_vals-CN_vals)

            plt.figure()
            plt.subplot(3,1,1)
            plt.imshow(np.real(mfe_vals),aspect='auto')
            #plt.clim([-2,2])
            plt.colorbar()
            plt.subplot(3,1,2)
            plt.imshow(np.real(CN_vals),aspect='auto')
            #plt.clim([-2,2])
            plt.colorbar()
            plt.subplot(3,1,3)
            plt.imshow(np.real(mfe_vals-CN_vals),aspect='auto')
            #plt.clim([-0.5,0.5])
            plt.colorbar()
            plt.title(str(tau))
            plt.savefig('plots/approxs_'+str(j)+'.pdf')
            plt.close()

            #plt.figure()
            #tt = np.linspace(0,T,Nt+1)
            #plt.semilogy(tt,np.linalg.norm(CN_vals,axis=0))
            #plt.semilogy(tt,10**(-5)*np.exp(rho/eps*tt),linestyle='dashed')
            #plt.savefig('plots/exponential_growth_CN'+str(j)+'.pdf')
            #plt.close()
 
            #plt.figure() 
            #plt.semilogy(rho**(np.abs(np.arange(-K,K+1,1))),linestyle='dashed')
            #plt.semilogy(np.array([1.0/np.sqrt(Nt*Nx)*np.linalg.norm(z_K[k*Nx:(k+1)*Nx,:]) for k in range(2*K+1)]))
            #plt.semilogy((rho*eps)**(np.abs(np.arange(-K,K+1,1))),linestyle='dashed')
            #plt.savefig('plots/z_Ks_rho_'+str(rho)+'_eps_'+str(eps)+'.pdf')
            #plt.close()
        print('MFE errs: ',mfe_errs)
        print('CN errs: ',cn_errs)
        # Ideal rho and eps
        perf[rhoind,epsind] = np.min(mfe_errs/(cn_errs+10**(-13)))
        print('Performance table:')
        print(perf)
        res = {'mfe_errs' : mfe_errs,
               'cn_errs'  : cn_errs,
               'rho'      : rho,
               'eps'      : eps,
               'taus'     : taus,
               'T'        : T
                }
        
        np.save('data/errs_rho_'+str(rho)+'_eps_'+str(eps)+'.npy',res,allow_pickle=True)
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
plt.imshow(perf, cmap='viridis')  # Heatmap
plt.colorbar()
plt.savefig('plots/perf.pdf')
       #breakpoint()
        #vals = np.array([f(xx,tau*j) for j in range(Nt+1)]).T
