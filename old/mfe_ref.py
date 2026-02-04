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
#
#Nx = 30
#Nt = 1000
#T  = 10
#tau = T/Nt
#
#Am = 20
#
#eps_mod = 0.1
#rho = 0.001
#
#norms = [0 for j in range(Am)]
#final_energy_defs = [0 for j in range(Am)]
#plt.figure()
#
def f(x,t):
    t0 = 2
    return np.exp(-20*x**2)*np.exp(-10*(t-t0)**2)
c = 100
def ui(x):
    return 0*np.exp(-c*(x-0)**2)
def vi(x):
    return 0*2*c*x*np.exp(-c*(x-0)**2)

def td_solver(f,eta,T,Nt,Nx,ui,vi,deg=40,return_sg = False,nx = 200):
    tau = T*1.0/Nt
    A,M,Bl,Br = spectral_galerkin_matrices(Nx,deg=deg)
    B = Bl + Br
    precomp = precomp_project(Nx,deg=deg)
    x_g,w   = leggauss(deg)
    bs = [project_legendre_from_values(f(x_g,tau*j),Nx,precomp=precomp,deg=deg,x=x_g) for j in range(Nt+1)]
       #return 1/(1.0+eps*np.cos(t/0.01))
    #usol[:,0] = project_legendre_from_values(np.exp(-x_g**2),Nx,precomp=precomp,deg=deg,x=x_g)
    sol = np.zeros((2*Nx,Nt+1))
    if ui is not None:
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
        LHS[:Nx,:Nx]     = 0.5*eta(tjp1)*A
        LHS[:Nx,Nx:2*Nx] = tau**(-1)*M 
        LHS[Nx:2*Nx,:Nx] = tau**(-1)*eye
        LHS[Nx:2*Nx,Nx:2*Nx] = -1.0/2*eye
    
        RHS[:Nx,:Nx]     = -0.5*eta(tj)*A
        RHS[:Nx,Nx:2*Nx] = tau**(-1)*M 
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
    xx = np.linspace(-1,1,nx)
    tt = np.linspace(0,T,Nt+1)
    vals = np.zeros((nx,Nt+1))
    #for j in range(Nt):
    #    #vals[:,j] = legval(xx,usol[:,j]) + (tau*j)**2*np.cos(xx)
    #    vals[:,j+1] = legval(xx,usol[:,j+1])
    vals = np.array([legval(xx,sol[:Nx,j]) for j in range(Nt+1)]).T
    print("VALS shape: ",vals.shape)
    #ex = np.array([(j*tau)**2*np.exp(-c*xx**2) for j in range(Nt+1)]).T
    #vals = vals-ex
    #from matplotlib.colors import LogNorm
    #plt.figure()
    ###plt.plot(xx,vals[:,-1])
    ###plt.plot(xx,T**2*np.exp(-c*xx**2),linestyle='dashed')
    #plt.imshow(vals,aspect = 0.1*(len(vals[0,:])/len(vals[:,0])), extent=[0,T,-1,1])
    #plt.savefig('ref.pdf')
    if return_sg:
        return vals,sol,M,A
    return vals

#vals = td_solver(f,eta,T,ui,vi)

#plt.show()
#plt.loglog(epss,np.abs(np.array(norms)-2.58466)) 
#plt.show()