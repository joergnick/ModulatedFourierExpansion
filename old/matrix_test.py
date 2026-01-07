import numpy as np

n = 4

A = np.zeros((n,n))
D = np.zeros((n,n))
for j in range(n-1):
    D[j,j] = 1.0/(j+1)
    A[j,j+1] = 1
    A[j+1,j] = 1

D[n-1,n-1] = 1.0/(n)
DA = D @ A

DA2 = DA @ DA 

print(DA)
print(DA2)
#
#for i in range(n):
#    for j in range(n):
#        if np.abs(A[i,j])>10**(-8):
#            A[i,j] = 1


#
#eps = 0.01
#rho = 0.1
#lam = 5
#s = 0.1+1j*10
#
#
#LHS = np.array([[(s-1j*1.0/eps)**2+lam , rho, 0], [rho,s**2+lam,rho],[0,rho,(s+1j*1.0/eps)**2+lam]])
#print(LHS)
#
#detLHS = ((s-1j*1.0/eps)**2+lam)*(s**2+lam)*((s+1j*1.0/eps)**2+lam)-rho**2*((s+1j*1.0/eps)**2+(s-1j*1.0/eps)**2+2*lam)
#
#def invert_3x3(A, tol=1e-12):
#    A = np.asarray(A, dtype=complex)
#    if A.shape != (3,3):
#        raise np.linalg.LinAlgError("Input must be a 3x3 matrix.")
#    a, b, c = A[0,0], A[0,1], A[0,2]
#    d, e, f = A[1,0], A[1,1], A[1,2]
#    g, h, i = A[2,0], A[2,1], A[2,2]
#
#    det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
#    if abs(det) <= tol:
#        raise np.linalg.LinAlgError("Matrix is singular (determinant is zero).")
#
#    # Build adjugate directly (transpose of cofactor matrix)
#    adj = np.array([
#        [ (e*i - f*h), -(b*i - c*h),  (b*f - c*e)],
#        [-(d*i - f*g),  (a*i - c*g), -(a*f - c*d)],
#        [ (d*h - e*g), -(a*h - b*g),  (a*e - b*d)]
#    ], dtype=complex).T  # transpose here to get adjugate
#
#    return adj / det,det
#
#
#
#invLHS,detLHSself = invert_3x3(LHS)
#print(np.linalg.norm(invLHS-np.linalg.inv(LHS)))
#print(np.linalg.det(LHS)-detLHS)
#print(np.linalg.det(LHS)-detLHSself)
#print(detLHS-detLHSself)
#
