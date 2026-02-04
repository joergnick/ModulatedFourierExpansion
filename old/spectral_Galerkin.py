import numpy as np
from numpy.polynomial.legendre import leggauss, legval, legder, legint

def spectral_galerkin_matrices(N,deg=40):
    x, w = leggauss(deg)
    # Compute right-hand side
    eye = np.eye(N)
    derind = [legder(eye[j]) for j in range(N)]
    #L1x = legval(x,eye[5])
    A = np.zeros((N,N)) 
    M = np.zeros((N,N)) 
    ## Setting the interior
    for j in range(N):
        weightedLj = w* legval(x,eye[j])
        M[j,:] = [np.sum(legval(x,eye[i])*weightedLj) for i in range(N)]
        weighteddLj = w* legval(x,derind[j])
        A[j,:] = [np.sum(legval(x,derind[i])*weighteddLj) for i in range(N)]
    ## Boundary condition
    Bleft = np.zeros((N,N))
    Bright = np.zeros((N,N))
    vleft = np.array([legval(-1,eye[j]) for j in range(N)] )
    #vleft = np.array([(-1)**(j) for j in range(N)] )
    vright = np.array([legval(1,eye[j]) for j in range(N)]) 
    #vright = np.array([1 for j in range(N)]) 
    for i in range(N):
        Bleft[i,:]  =  vleft*(-1)*vleft[i]
        Bright[i,:] = -vright*1 
        #B[i,:] = -(gammaright*vright*1 - gammaleft *vleft*(-1)*vleft[i])
    return A,M,Bleft,Bright
def complex_sg_matrices(N,deg=40):
    A,M,Bleft,Bright = spectral_galerkin_matrices(N,deg=deg)
    Ac = 1j *np.zeros((2*N,2*N))
    Mc = 1j *np.zeros((2*N,2*N))
    Bleftc = 1j *np.zeros((2*N,2*N))
    Brightc = 1j *np.zeros((2*N,2*N))
    Ac[:N,:N] = A
    Ac[N:,N:] = A
    Ac[:N,N:] = 1j*A
    Ac[N:,:N] = -1j*A

    Mc[:N,:N] = M
    Mc[N:,N:] = M
    Mc[:N,N:] = 1j*M
    Mc[N:,:N] = -1j*M

    Bleftc[:N,:N] = Bleft
    Bleftc[N:,N:] = Bleft
    Bleftc[:N,N:] = 1j*Bleft
    Bleftc[N:,:N] = -1j*Bleft

    Brightc[:N,:N] = Bright
    Brightc[N:,N:] = Bright
    Brightc[:N,N:] = 1j*Bright
    Brightc[N:,:N] = -1j*Bright
    return Ac,Mc,Bleftc,Brightc

def precomp_project(Nx,deg=40):
    x, w = leggauss(deg)
    eye = np.eye(Nx)
    precomp = [(2*j + 1)/2*w*legval(x,eye[j]) for j in range(Nx)]
    return np.array(precomp)

def project_legendre(func,N,deg=40,precomp=None,x=None):
    if x is None:
        x, w = leggauss(deg)
    fvals = func(x)
    if precomp is None:
        fweighted = w*fvals
        eye = np.eye(N)
        #b = np.array([np.sum(legval(x,eye[j])*fweighted) for j in range(N)])
        b = np.array([
            (2*j + 1)/2 * np.sum(legval(x, eye[j]) * fweighted)
            for j in range(N)
        ])
    else:
        #b = np.array([np.dot(precomp[j],fvals) for j in range(N) ])
        b = precomp.dot(fvals)
    return b

def project_legendre_from_values(fvals,N,deg=40,precomp=None,x=None):
    if x is None:
        x, w = leggauss(deg)
    if precomp is None:
        fweighted = w*fvals
        eye = np.eye(N)
        #b = np.array([np.sum(legval(x,eye[j])*fweighted) for j in range(N)])
        b = np.array([
            (2*j + 1)/2 * np.sum(legval(x, eye[j]) * fweighted)
            for j in range(N)
        ])
    else:
        #b = np.array([np.dot(precomp[j],fvals) for j in range(N) ])
        b = precomp.dot(fvals)
    return b

def sg_helmholtz_robin_solver(N,f,gammaleft,gammaright,deg=40):
    b = project_legendre(f,N,deg=deg)
    A,M,Bleft,Bright = spectral_galerkin_matrices(N,deg=40)
    sol_coeff = np.linalg.solve(A+gammaleft*Bleft+gammaright*Bright,b)
    return sol_coeff

def test_sg():
    def f(x):
        return np.exp(-20*x**2)
    deg = 40
    x_g,w   = leggauss(deg)
    Nx = 40
    precomp = precomp_project(Nx,deg=deg)
    #fcoeffs = project_legendre(f,Nx,deg=deg)
    fcoeffs = project_legendre_from_values(f(x_g),Nx,precomp=precomp,deg=deg,x=x_g)
    x_g = np.linspace(-1,1,100)
    f_ex = f(x_g)
    gammaleft = 100000000
    gammaright = 10000000
    
    sol_coeff = sg_helmholtz_robin_solver(40,f,gammaleft,gammaright,deg=40)
    vals = legval(x_g,sol_coeff)
    print(vals[0])
    print(vals[-1])
    import matplotlib.pyplot as plt
    plt.plot(np.real(vals))
    plt.savefig('test.png')
    return sol_coeff

test_sg()
