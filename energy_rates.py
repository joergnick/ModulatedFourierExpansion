import numpy as np
from spectral_Galerkin import spectral_galerkin_matrices,project_legendre_from_values,precomp_project,project_legendre
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor,lu_solve
from numpy.polynomial.legendre import leggauss, legval, legder

Nx = 30
Nt = 50000
T  = 2*np.pi
tau = T/Nt
Am = 10

Am_rho = 10
eps_mod = 0.4
epss = np.array([eps_mod*2**-j for j in range(Am)])
print("T/eps ",T/epss)
rhos = np.array([eps_mod*2**-j for j in range(Am_rho)])
norms = np.zeros((Am_rho,Am)) 
final_energy_defs = np.zeros((Am_rho,Am))
plt.figure()
 
deg = 80
A,M,Bl,Br = spectral_galerkin_matrices(Nx,deg=deg)
B = Bl + Br
usol = np.zeros((Nx,Nt+1))
vsol = np.zeros((Nx,Nt+1))
precomp = precomp_project(Nx,deg=deg)
x_g,w   = leggauss(deg)
def f(x,t):
    #gss = np.exp(-c*x**2)
    #return 2*gss + eta(t)*t**2*(2*c - (2*c*x)**2)*gss
    #return np.cos(t)/(1+t)*(0*x+1)
    return 0*x

c = 50
def ui(x):
    return np.exp(-c*(x-0)**2)
def vi(x):
    return 2*c*x*np.exp(-c*(x-0)**2)

bs = [project_legendre_from_values(f(x_g,tau*j),Nx,precomp=precomp,deg=deg,x=x_g) for j in range(Nt+1)]

for rho_ind in range(Am_rho):
    rho = rhos[rho_ind]
    for eps_ind in range(Am):
        eps = epss[eps_ind]
        def eta(t):
            #return 1.0
            return 1.0+rho*np.cos(t/eps)
            #return 1/(1.0+eps*np.cos(t/0.01))
       #usol[:,0] = project_legendre_from_values(np.exp(-x_g**2),Nx,precomp=precomp,deg=deg,x=x_g)
        sol = np.zeros((2*Nx,Nt+1))
        sol[:Nx,0]     = project_legendre(ui,Nx)
        sol[Nx:2*Nx,0] = project_legendre(vi,Nx)
        LHS = np.zeros((2*Nx,2*Nx))
        RHS = np.zeros((2*Nx,2*Nx))
        eye = np.eye(Nx)
        source = np.zeros((2*Nx,))
        for j in range(Nt):
            tj = j*tau
            tjp1 = tj+tau
            if j % 1000 == 0:
                1
                #print("Index: "+str(j))
            B=0*B
            LHS[:Nx,:Nx]     = 0.5*eta(tjp1)*A
            LHS[:Nx,Nx:2*Nx] = tau**(-1)*M - 0.5*eta(tjp1)*B
            LHS[Nx:2*Nx,:Nx] = tau**(-1)*eye
            LHS[Nx:2*Nx,Nx:2*Nx] = -1.0/2*eye
        
            RHS[:Nx,:Nx]     = -0.5*eta(tj)*A
            RHS[:Nx,Nx:2*Nx] = tau**(-1)*M + 0.5*eta(tj)*B
            RHS[Nx:2*Nx,:Nx] = tau**(-1)*eye
            RHS[Nx:2*Nx,Nx:2*Nx] = 0.5*eye
            source[:Nx] = 0.5*(bs[j]+bs[j+1]) 
            sol[:,j+1] = np.linalg.solve(LHS,np.matmul(RHS,sol[:,j])+source)
        print("Finished computation")
        nx = 100
        xx = np.linspace(-1,1,nx)
        tt = np.linspace(0,T,Nt+1)
        vals = np.zeros((nx,Nt+1))
        #for j in range(Nt):
        #    #vals[:,j] = legval(xx,usol[:,j]) + (tau*j)**2*np.cos(xx)
        #    vals[:,j+1] = legval(xx,usol[:,j+1])
        vals = np.array([legval(xx,sol[:Nx,j]) for j in range(Nt+1)]).T
        norms[rho_ind,eps_ind] = np.linalg.norm(vals[:,-1])
        print("VALS shape: ",vals.shape)
        energies = np.array([sol[Nx:,j].T @ M @ sol[Nx:,j]+eta(tau*j)*sol[:Nx,j].T @ A @ sol[:Nx,j] for j in range(len(sol[0,:]))])
        initial_energy = np.abs(sol[Nx:,0].T @ M @ sol[Nx:,0]+eta(tau*j)*sol[:Nx,0].T @ A @ sol[:Nx,0])
        final_energy = np.abs(sol[Nx:,-1].T @ M @ sol[Nx:,-1]+eta(tau*j)*sol[:Nx,-1].T @ A @ sol[:Nx,-1])
        #final_energy_defs[eps_ind]=np.abs(final_energy-initial_energy)
        final_energy_defs[rho_ind,eps_ind]=energies[-1]/energies[0]
        #final_energy_defs[rho_ind,eps_ind]=np.abs(energies[-1]-energies[0])
        #final_energy_defs[rho_ind,eps_ind]=np.max(np.abs(energies))-np.min(np.abs(energies))
        #energies += np.array([sol[Nx:,j].T @ M @ sol[Nx:,j] for j in range(len(sol[0,:]))])
        #plt.semilogy(tt, np.abs(energies-energies[0]) ,label=r"$\varepsilon=$"+str(eps))
        #plt.plot(tt, energies ,label=r"$\varepsilon=$"+str(eps))

#print(np.log(final_energy_defs)/T)
#plt.savefig('energies.pdf')
#plt.close()
rho_ov_eps = []
for rho_ind in range(Am_rho):
    for eps_ind in range(Am):
        rho_ov_eps.append(rhos[rho_ind]*1.0/epss[eps_ind])
rho_ov_eps = np.sort(rho_ov_eps)


import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
    "figure.dpi": 300,
    "text.latex.preamble" : r"\usepackage{amsmath}"  
})

# -----------------------
# Figure
# -----------------------
fig, ax = plt.subplots(figsize=(6.5, 5.5))

# Reference dashed line
plt.loglog(
    rho_ov_eps,
    rho_ov_eps,
    linestyle="--",
    color="black",
    linewidth=1.5,
    label=r"$\rho/\varepsilon$"
)

# Scatter data
for rho_ind in range(Am_rho):
    for eps_ind in range(Am):
        if (rho_ind == 0) and (eps_ind == 0):
            plt.scatter(
                rhos[rho_ind] / epss[eps_ind],
                np.log(final_energy_defs[rho_ind, eps_ind]) / T,
                s=40,
                alpha=0.85,
                edgecolors="blue",
                linewidths=0.4,
                color="tab:blue",
                label='Estimated growth rates'
            )
        else:
            plt.scatter(
                rhos[rho_ind] / epss[eps_ind],
                np.log(final_energy_defs[rho_ind, eps_ind]) / T,
                s=40,
                alpha=0.85,
                edgecolors="blue",
                linewidths=0.4,
                color="tab:blue"
            )
# -----------------------
# Axes formatting
# -----------------------
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e-6, 1e2)

ax.set_xlabel(r"$\rho / \varepsilon$")
ax.set_ylabel(r"$\log(E(T)-E(0))/T$")

ax.grid(which="both", linestyle=":", linewidth=0.6, alpha=0.7)
ax.legend(frameon=False)

#plt.tight_layout()

# -----------------------
# Save for publication
# -----------------------
plt.savefig("energy_scaling.pdf", bbox_inches="tight")
#plt.show()
#
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#import numpy as np
#
#mpl.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif",
#    "font.serif": ["Computer Modern Roman"],
#    "axes.labelsize": 18,
#    "xtick.labelsize": 14,
#    "ytick.labelsize": 14,
#    "legend.fontsize": 14,
#    "axes.linewidth": 1.2,
#    "figure.dpi": 300,
#})
#
#fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
#
## Reference line
#ax.loglog(
#    rho_ov_eps,
#    rho_ov_eps,
#    "--",
#    color="black",
#    linewidth=1.5,
#    label=r"$y=x$"
#)
#
## Scatter points (masked)
#for rho_ind in range(Am_rho):
#    for eps_ind in range(Am):
#        val = final_energy_defs[rho_ind, eps_ind]
#        if val > 0 and np.isfinite(val):
#            ax.scatter(
#                rhos[rho_ind] / epss[eps_ind],
#                np.log(val) / T,
#                s=40,
#                alpha=0.85,
#                edgecolors="black",
#                linewidths=0.4,
#                color="blue"
#            )
#
#ax.set_xscale("log")
#ax.set_yscale("log")
#ax.set_ylim(1e-3, 1e2)
#
#ax.set_xlabel(r"$\rho/\varepsilon$")
#ax.set_ylabel(r"$\log\!\left(E_{\mathrm{final}}\right)/T$")
#
#ax.grid(which="both", linestyle=":", linewidth=0.6)
#ax.legend(frameon=False)
#
#plt.savefig("energy_scaling.pdf", bbox_inches="tight")
#plt.show()
#
## -----------------------
## Save for publication
## -----------------------
#plt.savefig("energy_scaling.pdf", bbox_inches="tight")
#plt.show()
#

#plt.figure()
#plt.ylim([10**(-3),10**2])
#plt.loglog(rho_ov_eps,rho_ov_eps,linestyle='dashed')
##plt.loglog(rhos/eps_mod,rhos/eps_mod,linestyle='dashed')
##plt.loglog((rhos/eps_mod)**(-1),(rhos/eps_mod)**(-1),linestyle='dashed')
#for rho_ind in range(Am_rho):
#    for eps_ind in range(Am):
#        plt.scatter(
#            rhos[rho_ind]*1.0/epss[eps_ind],np.log(final_energy_defs[rho_ind,eps_ind])/T,
#            s=30,              # marker size
#            alpha=0.9,
#            linewidths=0.5,
#            edgecolors='blue',
#            color = 'blue'
#        )
#        #plt.scatter(rhos[rho_ind]*1.0/epss[eps_ind],np.log(final_energy_defs[rho_ind,eps_ind])/T,color = 'blue')
##plt.ylim([0,3])
##plt.legend()
#plt.savefig('energy_rates.pdf')
##plt.show()
#plt.figure()
#plt.imshow(np.log(final_energy_defs)/T)
#plt.savefig('energies_rho_eps.pdf')
##plt.figure()
##plt.loglog(epss,final_energy_defs)
##plt.loglog(epss,epss,linestyle='dashed')
##plt.loglog(epss,(epss/eps_mod)**2,linestyle='dashed')
###plt.ylim([0,3])
###plt.legend()
##plt.savefig('energy_diffs.pdf')
###plt.show()
#    
##plt.loglog(epss,np.abs(np.array(norms)-2.58466)) 
#plt.show()