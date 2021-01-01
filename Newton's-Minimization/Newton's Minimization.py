import numpy as np
from sympy import *
def hessian(f,xi):
  symb=[k for k in [x,y,z,v] if f.count(k)!=0]
  n=len(symb)
  M=np.zeros((n,n))
  for p in range(0,n):
    for j in range(0,n):
      holder=diff(f,symb[p],symb[j])
      for q in range(0,n):
        holder=holder.subs(symb[q],xi[q])
      M[p][j]=holder
  M=np.linalg.inv(M)
  return M
def grad(f,xi):
  symb=[k for k in [x,y,z,v] if f.count(k)!=0]
  n=len(symb)
  M=np.zeros(n)
  for p in range(0,n):
    holder=diff(f,symb[p])
    for q in range(0,n):
      holder=holder.subs(symb[q],xi[q])
    M[p]=holder
  M=M.reshape(n,1)
  return M
def Newton_min(f,tol=10e-5):
  symb=[k for k in [x,y,z,v] if f.count(k)!=0]
  n=len(symb)
  x0=np.array([28,10,6]).reshape(n,1)# initial value
  i=0
  while True:
    hess=hessian(f,x0)        
    gradient=grad(f,x0)       
    xk=x0-np.dot(hess,gradient)  
    error=np.linalg.norm(x0-xk)
    x0=xk
    if i>200:
      break
    if error<tol:
      break
    i+=1
  return x0,i


x,y,z,v=symbols('x y z v')       
f=(x-30)**4+2*(y-12)**2+80*(z-5)**6           
X,i=Newton_min(f)
print("Iteration ",i)
print("The minima point is : ",X)
