import numpy as np
import scipy
import sys
sys.path.append('./cqToolbox')
from numpy.polynomial.legendre import leggauss, legval, legder
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import LinearOperator,spilu,gmres

from finite_difference_cn_helpers import finite_difference_matrices
import matplotlib.pyplot as plt
# np.sort(np.abs(np.linalg.eigvals(np.linalg.inv(LHS_csc.toarray()@RHS_csc))))

def make_mfe_sol(rho,eps,Nt,T,Nx,K,f,deg,xx,return_sg=False):
    tau = T*1.0/Nt
    A,M,Bl,Br = finite_difference_matrices(Nx)
    #A = A.toarray()
    #M = M.toarray()
    ## Build system
    #D_K = 1j*np.zeros(((2*K+1)*Nx,(2*K+1)*Nx))
    #M_K = 1j*np.zeros(((2*K+1)*Nx,(2*K+1)*Nx))
    #T_A = 1j*np.zeros(((2*K+1)*Nx,(2*K+1)*Nx))
    D_K = lil_matrix(((2*K+1)*Nx,(2*K+1)*Nx),dtype=np.complex128)
    M_K = lil_matrix(((2*K+1)*Nx,(2*K+1)*Nx),dtype=np.complex128)
    T_A = lil_matrix(((2*K+1)*Nx,(2*K+1)*Nx),dtype=np.complex128)
     
    for j in range(2*K+1):
        T_A[j*Nx:(j+1)*Nx,j*Nx:(j+1)*Nx] = A
        D_K[j*Nx:(j+1)*Nx,j*Nx:(j+1)*Nx] = 1j*(j-K)/eps*M
        M_K[j*Nx:(j+1)*Nx,j*Nx:(j+1)*Nx] = M
    for j in range(2*K):
        T_A[j*Nx:(j+1)*Nx,(j+1)*Nx:(j+2)*Nx] = rho * A
        T_A[(j+1)*Nx:(j+2)*Nx,j*Nx:(j+1)*Nx] = rho * A

    #LHS = 1j*np.zeros((2*(2*K+1)*Nx,2*(2*K+1)*Nx))
    #RHS = 1j*np.zeros((2*(2*K+1)*Nx,2*(2*K+1)*Nx))
    LHS = lil_matrix((2*(2*K+1)*Nx,2*(2*K+1)*Nx),dtype=np.complex128)
    RHS = lil_matrix((2*(2*K+1)*Nx,2*(2*K+1)*Nx),dtype=np.complex128)
     
    b_ind = (2*K+1)*Nx
    LHS[:b_ind,:b_ind] = 0.5*(T_A)
    LHS[:b_ind,b_ind:] = 1.0/tau*M_K+0.5*D_K
    LHS[b_ind:,:b_ind] = 1.0/tau*M_K+0.5*D_K
    LHS[b_ind:,b_ind:] = -1.0/2*M_K
    
    RHS[:b_ind,:b_ind] = -0.5*(T_A)
    RHS[:b_ind,b_ind:] = 1.0/tau*M_K-0.5*D_K
    RHS[b_ind:,:b_ind] = 1.0/tau*M_K-0.5*D_K
    RHS[b_ind:,b_ind:] = 1.0/2*M_K
    
    RHS[::Nx,:] = 0
    RHS[:,::Nx] = 0

    LHS[::Nx,:] = 0
    LHS[::Nx,:] = 0

    RHS[Nx-1::Nx,:] = 0
    RHS[:,Nx-1::Nx] = 0

    LHS[Nx-1::Nx,:] = 0
    LHS[Nx-1::Nx,:] = 0
 
    print("System assembled.")
    for k in range(2*(2*K+1)):
        RHS[k*Nx,k*Nx]=1
        LHS[k*Nx,k*Nx]=1
    for k in range(2*(2*K+1)):
        RHS[Nx-1+k*Nx,Nx-1+k*Nx]=1
        LHS[Nx-1+k*Nx,Nx-1+k*Nx]=1
    L = 1
    x_g   = np.linspace(0,L,Nx)
    fvals = [M @ f(x_g,tau*j) for j in range(Nt+1)]
     
    z_K = 1j*np.zeros((2*(2*K+1)*Nx,Nt+1))
    rhs = np.array(fvals).T

    source  = np.zeros((2*(2*K+1)*Nx,))
    #P,L_LHS,R_LHS = scipy.linalg.lu(LHS)
    LHS_csc = LHS.tocsc()
    RHS_csc = RHS.tocsc()
    ilu = spilu(LHS_csc)
    prec = LinearOperator(
        shape = LHS_csc.shape,
        matvec = lambda x: ilu.solve(x)
        )
    print("Preconditioner computed.")
    #P,L_LHS,R_LHS = scipy.linalg.lu(LHS)
    #source  = np.zeros((2*(2*K+1)*Nx,Nt+1))
    #np.sort(np.abs(np.linalg.eigvals(np.linalg.inv(LHS_csc.toarray()@RHS_csc))))
    #Is = np.argsort(np.abs(np.linalg.eigvals(np.linalg.inv(LHS_csc.toarray()@RHS_csc))))
    for j in range(Nt):
        tj = j*tau
        tjp1 = tj+tau
        source[K*Nx:(K+1)*Nx] = 0.5*rhs[:,j]+0.5*rhs[:,j+1]
        #source[(K-1)*Nx:(K)*Nx] = 0.5*rhs[:,j]+0.5*rhs[:,j+1]
        #source[(K+1)*Nx:(K+2)*Nx] = 0.5*rhs[:,j]+0.5*rhs[:,j+1]
        if j % 1000 == 0:
            1
            #print("Index: "+str(j))
        #z_K[:,j+1] = scipy.sparse.linalg.spsolve(LHS.tocsr(), RHS @ z_K[:,j]+source)
        z_K[:,j+1],info = gmres(LHS_csc, RHS_csc @ z_K[:,j]+source,M=prec,tol=1e-10,maxiter=200)

        #z_K[0::Nx,j+1]    = 0
        #z_K[Nx-1::Nx,j+1] = 0

        #z_K[:,j+1],info = gmres(LHS_csc, RHS_csc @ z_K[:,j]+source,M=prec,tol=1e-8)
        if info != 0:
            print("Linear Algebra issue, breaking.")
            z_K[:,j+1] = scipy.sparse.linalg.spsolve(LHS_csc, RHS_csc @ z_K[:,j]+source)

        #z_K[:,j+1] = scipy.linalg.solve_triangular(L_LHS , P@(np.matmul(RHS,z_K[:,j])+source),lower=True)
        #z_K[:,j+1] = scipy.linalg.solve_triangular(R_LHS , z_K[:,j+1])
        #z_K[:,j+1] = np.linalg.solve(LHS,np.matmul(RHS,z_K[:,j])+source)

        if np.linalg.norm(z_K[:,j+1])>10**10:
           #plt.semilogy(range(195,205),np.abs(z_K[195:205,j+1]))
           #plt.semilogy(range(195,205),np.abs(z_K[195:205,j+1]))
            plt.semilogy(np.abs(z_K[:(2*K+1)*Nx,j+1]))
            plt.ylim([1e-16,1e4])
            plt.savefig("z_K.pdf")
           # np.linalg.norm(LHS_csc@z_K[:,j+1]-RHS_csc@z_K[:,j]-source)
            plt.figure(12)
            plt.spy(T_A)
            plt.savefig("TA.pdf")
            plt.figure(100)
            plt.spy(RHS)
            plt.savefig("RHS.pdf")

#           plt.semilogy(np.abs(v_mx))
#           plt.semilogy(np.abs(It_v_mx))
#           plt.savefig("worst_mode.pdf")
            break
    z_K=z_K[:(2*K+1)*Nx,:]

    xx = x_g
    tt = np.linspace(0,T,Nt+1)
    #vals = np.zeros((Nx,Nt+1))
    mfe_sol = 1j*np.zeros((Nx,Nt+1))
    for j in range(Nt+1):
        tj = tt[j]
        for k_ind in range(2*K+1):
            k_ = k_ind-K
            mfe_sol[:,j]  += np.exp(1j*k_*tj/eps)*z_K[k_ind*Nx:(k_ind+1)*Nx, j]
    mfe_vals = np.array([mfe_sol[:,j] for j in range(Nt+1)]).T
    if return_sg:
        return mfe_vals,z_K,mfe_sol
    return mfe_vals,z_K
