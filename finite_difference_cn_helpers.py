import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse.linalg
from scipy.sparse import lil_matrix
#--------------------------------------------------------------------
# 1. FEM mass and stiffness matrices
#--------------------------------------------------------------------
import numpy as np
from scipy.sparse import lil_matrix

def mass_matrix(N, h):
    """
    Assemble 1D finite element mass matrix in LIL format
    """
    M = lil_matrix((N, N))
    
    # Element mass matrix
    Me = (h / 6) * np.array([[2, 1],
                             [1, 2]])
    
    # Assemble global matrix
    for e in range(N - 1):
        nodes = [e, e + 1]
        M[np.ix_(nodes, nodes)] += Me
    
    # Apply Dirichlet boundary conditions
    M[0, :] = 0
    M[-1, :] = 0
    M[:, 0] = 0
    M[:, -1] = 0
    M[0, 0] = 1
    M[-1, -1] = 1
    return M.toarray()

#def mass_matrix(N, h):
#    M = np.zeros((N, N))
#    Me = (h/6) * np.array([[2, 1],
#                           [1, 2]])
#    for e in range(N-1):
#        nodes = [e, e+1]
#        M[np.ix_(nodes, nodes)] += Me
#    M[0,:] = M[-1,:] = 0
#    M[:,0] = M[:,-1] = 0
#    M[0,0] = M[-1,-1] = 1
#    return M


#def stiffness_matrix(N, h):
#    A = np.zeros((N, N))
#    Ae = (1/h) * np.array([[1, -1],
#                           [-1, 1]])
#    for e in range(N-1):
#        nodes = [e, e+1]
#        A[np.ix_(nodes, nodes)] += Ae
#    A[0,:] = A[-1,:] = 0
#    A[:,0] = A[:,-1] = 0
#    A[0,0] = A[-1,-1] = 1
#    return A

def stiffness_matrix(N, h):
    A = lil_matrix((N, N))
    #Ae = (1/h) * np.array([[1, -1],
    #                       [-1, 1]])
    for e in range(N - 1):
        #nodes = [e, e + 1]
        #A[np.ix_(nodes, nodes)] += Ae
        A[e, e]     += 1/h
        A[e, e + 1] -= 1/h
        A[e + 1, e] -= 1/h
        A[e + 1, e + 1] += 1/h
    A[0, :] = 0
    A[-1, :] = 0
    A[:, 0] = 0
    A[:, -1] = 0

    A[0, 0] = 1
    A[-1, -1] = 1

    #return A
    return A.toarray()


def finite_difference_matrices(Nx):
    L = 1
    x = np.linspace(0, L, Nx)
    h = x[1] - x[0]
    A = stiffness_matrix(Nx,h)
    M = mass_matrix(Nx,h)
    return A,M,0*A,0*A
#--------------------------------------------------------------------
# 2. Crank–Nicolson stepping with c(x,t)
#--------------------------------------------------------------------
def crank_nicolson_step_variable_c(h, c_func, dt, Nt, u0, v0, x):
    Nx = len(u0)

    M = mass_matrix(Nx, h)
    K = stiffness_matrix(Nx, h)
    Z = np.zeros((Nx, Nx))
    
    u = u0.copy()
    v = v0.copy()
    
    z = np.concatenate([u, v])
    
    for n in range(Nt):
        t = n*dt
        cvec = c_func(x, t)         # c at nodes at current time
        C2 = np.diag(cvec**2)       # diagonal matrix
        
        A_big = np.block([[ Z,      M          ],
                          [ -K @ C2, Z        ]])
        I_big = np.block([[ M, Z],
                          [ Z, M ]])
        
        A = I_big - 0.5*dt * A_big
        B = I_big + 0.5*dt * A_big
        
        z = np.linalg.solve(A, B @ z)
    
    u = z[:Nx]
    v = z[Nx:]
    return u, v
def poisson_test():
    start = time.time()
    Nx = 200
    A,M,_,_ = finite_difference_matrices(Nx)
    xx = np.linspace(0,1,Nx)
    def f(x):
        return 0*x+2
    def u(x):
        return x*(1-x)
    b = f(xx)
    rhs = M @ f(xx)
    rhs[0] = rhs[-1] = 0
    
    #u_sol = np.linalg.solve(A,rhs)
    u_sol = scipy.sparse.linalg.spsolve(A,rhs)
    end = time.time()
    print('Duration: '+str(end-start))
    import matplotlib.pyplot as plt
    plt.plot(u_sol)
    plt.plot(u(xx),linestyle='dashed')
    plt.savefig('test_poisson.png')
    return 
#poisson_test()