import numpy as np
import scipy
import sys
sys.path.append('./cqToolbox')
#from spectral_Galerkin import spectral_galerkin_matrices,project_legendre_from_values,precomp_project,project_legendre
from finite_difference_cn_helpers import finite_difference_matrices
from numpy.polynomial.legendre import leggauss, legval, legder
import matplotlib.pyplot as plt
from cqToolbox.linearcq import Conv_Operator

def time_harmonic_solve(s,f_hat,K,Nx,rho,eps,precomp=None):
    if precomp is None:
        A,M,Bl,Br = finite_difference_matrices(Nx,deg=50)
    else:
        A,M,Bl,Br = precomp[0],precomp[1],precomp[2],precomp[3]
    # Create T_A
    D_K = 1j*np.zeros(((2*K+1)*Nx,(2*K+1)*Nx))
    T_A = 1j*np.zeros(((2*K+1)*Nx,(2*K+1)*Nx))
    for j in range(2*K+1):
        T_A[j*Nx:(j+1)*Nx,j*Nx:(j+1)*Nx] = A
        D_K[j*Nx:(j+1)*Nx,j*Nx:(j+1)*Nx] = (s+1j*(j-K)/eps)**2*M
    for j in range(2*K):
        T_A[j*Nx:(j+1)*Nx,(j+1)*Nx:(j+2)*Nx] = rho * A
        T_A[(j+1)*Nx:(j+2)*Nx,j*Nx:(j+1)*Nx] = rho * A
    LHS = D_K+T_A
    x_hat = np.linalg.solve(LHS,f_hat)
    return x_hat


def make_mfe_sol(rho,eps,Nt,T,Nx,K,f,deg,xx,return_sg=False):
    tau = T*1.0/Nt
    #precomp_pr = precomp_project(Nx,deg=deg)
    precomp_fd = finite_difference_matrices(Nx)
    rhs = create_rhs(Nt,T,Nx,K,f,precomp = precomp_fd,deg=deg)
    def th_sys(s,b):
        x_hat = time_harmonic_solve(s,b,K,Nx,rho,eps,precomp=precomp_fd)
        return x_hat
    td_sys = Conv_Operator(th_sys,order=-1)
    # CAREFUL OF INITIAL VALUES
    z_K = td_sys.apply_convol_no_symmetry(rhs,T,show_progress = False,cutoff = 10**(-9))
    mfe_sol = 1j*np.zeros((Nx,Nt+1))
    for j in range(Nt+1):
        for k_ind in range(2*K+1):
            k = k_ind-K
            mfe_sol[:,j]  += np.exp(1j*k*tau*j/eps)*z_K[k_ind*Nx:(k_ind+1)*Nx, j]
    mfe_vals = np.array([mfe_sol[:,j] for j in range(Nt+1)]).T
    if return_sg:
        return mfe_vals,z_K,mfe_sol
    return mfe_vals,z_K

def create_rhs(Nt,T,Nx,K,f,precomp = None,deg=50):
    if precomp is None:
        precomp = finite_difference_matrices(Nx)
    L = 1
    x_g = np.linspace(0,L,Nx)
    rhs = np.zeros(((2*K+1)*Nx,Nt+1))
    tau = T*1.0/Nt
    fvals = [precomp[1] @ f(x_g,tau*j) for j in range(Nt+1)]
    rhs[(K)*Nx:(K+1)*Nx,:] = np.array(fvals).T
    return rhs
