import numpy as np
import numdifftools as nd
from numpy.linalg import norm

def Steepest_Descent(x,eps=10e-8):

    i = 0 
    while True:
        grad = nd.Gradient(func)(x[i]) 
        grad_norm = norm(grad,2)      
        if grad_norm < eps:            
            print(f"minimum point {x[i]} is obtained at iteration {i}")
            break
        _dir = -grad                   
        H = nd.Hessian(func)(x[i])     
        alpha = (_dir.T @ _dir) / (_dir.T @ H @ _dir) 
               
        new_x = x[i] + alpha * _dir   
        x.append(new_x)               
       
        i+=1                         
    
    return

def func(x): 
    return ( ((x[0] - 2)**2)/8 + ((x[1]-3)**2)/12  ) 

x = []                  
x.append([1,1])      
Steepest_Descent(x)  
