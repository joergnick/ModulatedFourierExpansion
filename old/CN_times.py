import numpy as np
from spectral_Galerkin import spectral_galerkin_matrices,project_legendre_from_values,precomp_project,project_legendre
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor,lu_solve
from numpy.polynomial.legendre import leggauss, legval, legder
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

Nx = 30
Nt = 1000000
Am = 1

epss = [0.1*2**-j for j in range(Am)]
norms = [0 for j in range(Am)]
plt.figure()
Am_alph = 12
alphas = np.array([2**(j) for j in range(Am_alph)])
diffs = [0 for j in range(Am_alph)]
mod = True
for alpha_ind in range(Am_alph):
    alpha = alphas[alpha_ind]
    T  = alphas[alpha_ind]
    tau = T/Nt
    for eps_ind in range(Am):
        eps = epss[eps_ind]
        def eta(t):
            #return 1.0
            return 1.0+eps**(1.5)*np.sin(t/eps)
            #return 1/(1.0+eps**(1+alpha)*np.sin(t/eps))
        
        deg = 90
        A,M,Bl,Br = spectral_galerkin_matrices(Nx,deg=deg)
        
        B = Bl + Br
        usol = np.zeros((Nx,Nt+1))
        vsol = np.zeros((Nx,Nt+1))
        precomp = precomp_project(Nx,deg=deg)
        x_g,w   = leggauss(deg)
        #usol[:,0] = project_legendre_from_values(np.exp(-x_g**2),Nx,precomp=precomp,deg=deg,x=x_g)
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
        sol = np.zeros((2*Nx,Nt+1))
        sol[:Nx,0] = project_legendre(ui,Nx)
        sol[Nx:2*Nx,0] = project_legendre(vi,Nx)
        LHS = np.zeros((2*Nx,2*Nx))
        RHS = np.zeros((2*Nx,2*Nx))
        eye = np.eye(Nx)
        source = np.zeros((2*Nx,))
        for j in range(Nt):
            tj = j*tau
            tjp1 = tj+tau
            if j % 1000 == 0:
                print("Index: "+str(j))
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
        norms[eps_ind] = np.linalg.norm(vals[:,-1])
        print("VALS shape: ",vals.shape)
        energies = np.array([eta(tau*j)**(1)*sol[:Nx,j].T @ A @ sol[:Nx,j] for j in range(len(sol[0,:]))])
        energies += np.array([sol[Nx:,j].T @ M @ sol[Nx:,j] for j in range(len(sol[0,:]))])
        #plt.plot(tt, energies ,label=r"$\varepsilon=$"+str(eps))
        diffs[alpha_ind] = max(energies)- min(energies)
plt.loglog(alphas,diffs,label=r"$\max\mathcal{E}-\min\mathcal{E}$")
#plt.loglog(alphas,np.array(alphas)**(1),linestyle='dashed',label=r"$T$")
#plt.loglog(epss,np.array(epss)**2,linestyle='dashed',label='eps')
#plt.loglog(epss,np.array(epss)**(2+2*alpha),linestyle='dashed',label='3alpha')
#plt.loglog(epss,np.array(epss)**(2+3*alpha),linestyle='dashed',label='3alpha')
#plt.ylim([0,2])
if mod:
    plt.title(r"Modulated energy $\mathcal{E} (t) = \|\dot{u}(t)\|^2+\kappa(t) \|\nabla u(t) \|^2$")
else:
    plt.title(r"Unmodulated energy $\mathcal{E} (t) = \|\dot{u}(t)\|^2+ \|\nabla u(t) \|^2$")
plt.xlabel(r"$T$")
plt.legend()

if mod:
    plt.savefig('modulated_energy.pdf')
else:
    plt.savefig('unmodulated_energy.pdf')

plt.show()
 