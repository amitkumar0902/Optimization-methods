import numpy as np
A=np.array([[8,1],[1,2]])
#print(Q)
b=np.array([1,3])
x=np.array([0,0])

def CJ(A,b,x):
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(np.transpose(r), r)
    for i in range(len(b)):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(np.transpose(r), r)
        if np.sqrt(rsnew) < 1e-12:
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
        #print(x)
    return x

x=CJ(A,b,x)
print("solution is ",x)
