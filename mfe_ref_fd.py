import numpy as np
from finite_difference_cn_helpers import finite_difference_matrices
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import scipy.sparse.linalg
from scipy.linalg import lu_factor,lu_solve
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
#def f(x,t):
#    t0 = 2
#    return np.exp(-20*x**2)*np.exp(-10*(t-t0)**2)

#c = 100
#def ui(x):
#    return 0*np.exp(-c*(x-0)**2)
#def vi(x):
#    return 0*2*c*x*np.exp(-c*(x-0)**2)
#def f(x,t):
#    t0 = 2
#    return np.exp(-10000*(x-0.5)**2)*np.exp(-20000*(t-t0)**2)

def td_solver(f,eta,T,Nt,Nx,ui,vi,deg=40,return_sg = False,nx = 200):
    tau = T*1.0/Nt
    A,M,Bl,Br = finite_difference_matrices(Nx)
    B = Bl + Br
    L = 1
    x_g   = np.linspace(0,L,Nx)
    bs = [M @ f(x_g,tau*j) for j in range(Nt+1)]
       #return 1/(1.0+eps*np.cos(t/0.01))
    #usol[:,0] = project_legendre_from_values(np.exp(-x_g**2),Nx,precomp=precomp,deg=deg,x=x_g)
    sol = np.zeros((2*Nx,Nt+1))
    if ui is not None:
        sol[:Nx,0]     = ui(x_g)
        sol[Nx:2*Nx,0] = vi(x_g)
    LHS = lil_matrix((2*Nx, 2*Nx))
    RHS = lil_matrix((2*Nx, 2*Nx))
    #LHS = np.zeros((2*Nx,2*Nx))
    #RHS = np.zeros((2*Nx,2*Nx))
    source = np.zeros((2*Nx,))
    LHS[:Nx,Nx:2*Nx] = tau**(-1)*M 
    LHS[Nx:2*Nx,:Nx] = -tau**(-1)*M
    LHS[Nx:2*Nx,Nx:2*Nx] = 1.0/2*M

    RHS[:Nx,Nx:2*Nx] = tau**(-1)*M 
    RHS[Nx:2*Nx,:Nx] = -tau**(-1)*M
    RHS[Nx:2*Nx,Nx:2*Nx] = -0.5*M
    for j in range(Nt):
        tj = j*tau
        tjp1 = tj+tau
        if j % 1000 == 0:
            1
            #print("Index: "+str(j))
        LHS[:Nx,:Nx]     = 0.5*eta(tjp1)*A
        RHS[:Nx,:Nx]     = -0.5*eta(tj)*A
        source[:Nx] = 0.5*(bs[j]+bs[j+1]) 
        sol[:,j+1] = scipy.sparse.linalg.spsolve(LHS.tocsr(), RHS @ sol[:,j]+source)
       # if j==Nt-1:
       #     breakpoint()
        #sol[:,j+1] = np.linalg.solve(LHS,np.matmul(RHS,sol[:,j])+source)
    xx = x_g
    tt = np.linspace(0,T,Nt+1)
    #vals = np.zeros((Nx,Nt+1))
    vals = sol[:Nx,:]
    vals[0,:]=0
    vals[-1,:]=0
 
    #vals = np.array([sol[:Nx,j] for j in range(Nt+1)]).T
    if return_sg:
        return vals,sol,M,A
    return vals
#
#import time

#start = time.time()
#rho = 0.1
#eps = 0.1
#def eta(t):
#    return 1+2*rho*np.cos(t/eps)
#vals = td_solver(f,eta,5,200,200,ui,vi)
#end = time.time()
#print('duration: ',end-start)

#import matplotlib.pyplot as plt
#plt.imshow(np.real(vals),aspect='auto')
#plt.colorbar()
#plt.savefig('approxs.pdf')
#
