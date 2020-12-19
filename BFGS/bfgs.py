import numpy as np
import numpy.linalg as ln
import scipy as sp
import scipy.optimize
import numdifftools as nd
# Objective function
def f(x):
    return 4*(x[0]**2) + (x[0] * x[1])+ x[1]**2 - x[0] - 3*x[1] +7

def bfgs_method(f, x0, maxiter=None, epsi=10e-12):
    # Derivative
    fprime=nd.Gradient(f)
    print(fprime)
    if maxiter is None:
        maxiter = len(x0) * 200
    # initial values
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
        k += 1
        ro = 1.0 / (np.dot(yk, sk))
        A1 = I - ro * sk[:, np.newaxis] * yk[np.newaxis, :]
        A2 = I - ro * yk[:, np.newaxis] * sk[np.newaxis, :]
        Hk = np.dot(A1, np.dot(Hk, A2)) + (ro * sk[:, np.newaxis] * sk[np.newaxis, :])
    return (xk, k)
result, k = bfgs_method(f, np.array([0,0]))
print('The solution of bfgs algorithm is %s' % (result))
print('The number of iterations is %s' % (k))
