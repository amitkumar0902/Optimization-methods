def f(x,Q,b,c):
    return (0.5*(x.T).dot(Q.dot(x))+(x.T).dot(b)+c)

def grad_f(x,Q,b,c):
    return (Q.dot(x)+b)

def SteepestDescent(x,Q,b,c):
    while True:
        temp=x
        g=grad_f(x,Q,b,c)
        alpha=((g.T).dot(g))/((g.T).dot(Q.dot(g)))
        x=x-alpha*grad_f(x,Q,b,c)
        if (np.linalg.norm(x-temp)<1e-12):
            break
    return x
x = np.array([[0.0],[0.0]])
Q = np.array([[2.0, 5.0], [5.0, 15.0]])
b = np.array([[1.0], [2.0]])  
c = 8
x=SteepestDescent(x,Q,b,c)
print("The optimum point is : ",x)
