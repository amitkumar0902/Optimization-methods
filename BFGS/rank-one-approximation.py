import numpy as np
import numpy.linalg as ln
import scipy as sp
import scipy.optimize
import numdifftools as nd

# Objective function
def f(x):
    

def rank_one_method(f, x0, maxiter=None, epsi=10e-6):
    fprime=nd.Gradient(f)
    if maxiter is None:
        maxiter = len(x0) * 200
    k = 0
    gfk = fprime(x0)
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I
    xk = x0
    while ln.norm(gfk) > epsi and k < maxiter:
        pk = -np.dot(Hk, gfk)
        line_search = sp.optimize.line_search(f, fprime, xk, pk)
        alpha_k = line_search[0]
        xkp1 = xk + alpha_k * pk
        sk = xkp1 - xk #delta X
        xk = xkp1
        gfkp1 = fprime(xkp1)
        yk = gfkp1 - gfk  #delta gradient
        gfk = gfkp1
        
        u = sk - Hk @ yk
        Hk = Hk + (u @ u.T)/(u.T @ yk)
        
        k += 1
    return (xk, k)


result, k = rank_one_method(f, np.array([0, 0]), 2)

print('minima point: %s' % (result))
