import numpy as np
from spectral_Galerkin import spectral_galerkin_matrices,project_legendre_from_values,precomp_project,project_legendre
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor,lu_solve
from numpy.polynomial.legendre import leggauss, legval, legder
import matplotlib.pyplot as plt
#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "Helvetica"
#})

Nx = 30
Nt = 10000
T  = 10*2*np.pi
tau = T/Nt

Am = 20

eps_mod = 0.01
epss = np.array([eps_mod*1.5**-j for j in range(Am)])

norms = [0 for j in range(Am)]
final_energy_defs = [0 for j in range(Am)]
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

for eps_ind in range(Am):
    eps = epss[eps_ind]
    def eta(t):
        #return 1.0
        return 1.0+eps*np.cos(t/eps_mod)
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
        #from scipy.linalg import svdvals
        #e = svdvals(np.abs(LHS))
        #print(min(e))
        #plt.semilogy(np.sort(np.abs(e)))
        #plt.show()
        #breakpoint()
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
    #ex = np.array([(j*tau)**2*np.exp(-c*xx**2) for j in range(Nt+1)]).T
    #vals = vals-ex
    #from matplotlib.colors import LogNorm
    #plt.figure()
    ###plt.plot(xx,vals[:,-1])
    ###plt.plot(xx,T**2*np.exp(-c*xx**2),linestyle='dashed')
    #plt.imshow(np.abs(vals),aspect = 0.1*(len(vals[0,:])/len(vals[:,0])), extent=[0,T,-1,1])
    #plt.colorbar()
    #plt.show()
    #plt.figure(2)
    ##sol = legval(xx,legval(xx,bj))
    ###plt.figure()
    ####plt.plot(xx,sol)
    ####plt.loglog(tt[10:],np.linalg.norm(usol,axis=0)[10:])
    ###plt.loglog(tt[1:],np.linalg.norm(vals,axis=0)[1:],label='$|u(t)|_2$')
    ###plt.loglog(tt[1:],0.5*np.power(tt[1:],1.0/2),linestyle='dashed',label='$ O(\sqrt{t})$')
    ###plt.loglog(tt[1:],0.5*tt[1:],linestyle='dashed',label='$ O(t)$')
    ###plt.legend()
    energies = np.array([sol[Nx:,j].T @ M @ sol[Nx:,j]+eta(tau*j)*sol[:Nx,j].T @ A @ sol[:Nx,j] for j in range(len(sol[0,:]))])
    initial_energy = np.abs(sol[Nx:,0].T @ M @ sol[Nx:,0]+eta(tau*j)*sol[:Nx,0].T @ A @ sol[:Nx,0])
    final_energy = np.abs(sol[Nx:,-1].T @ M @ sol[Nx:,-1]+eta(tau*j)*sol[:Nx,-1].T @ A @ sol[:Nx,-1])
    #final_energy_defs[eps_ind]=np.abs(final_energy-initial_energy)
    final_energy_defs[eps_ind]=np.max(np.abs(energies))-np.min(np.abs(energies))
    #energies += np.array([sol[Nx:,j].T @ M @ sol[Nx:,j] for j in range(len(sol[0,:]))])
    plt.semilogy(tt, np.abs(energies-energies[0]) ,label=r"$\varepsilon=$"+str(eps))
    #plt.plot(tt, energies ,label=r"$\varepsilon=$"+str(eps))

plt.savefig('energies.pdf')
plt.close()
plt.figure()
plt.loglog(epss,final_energy_defs)
plt.loglog(epss,epss,linestyle='dashed')
plt.loglog(epss,(epss/eps_mod)**2,linestyle='dashed')
#plt.ylim([0,3])
#plt.legend()
plt.savefig('energy_diffs.pdf')
#plt.show()
    
#plt.loglog(epss,np.abs(np.array(norms)-2.58466)) 
#plt.show()