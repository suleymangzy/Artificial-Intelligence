import numpy as np

def lineerFunction(x,w,b):
    f = w * x + b
    return f

def costFunction(y,x,w,b,m):
    J = (1/(2*m)) * np.sum((lineerFunction(x,w,b)-y)**2)
    return J

def gradientDescent(x,y,w,b,m,a):
    dw = (1/m) * np.sum((lineerFunction(x,w,b) - y) * x)
    db = (1/m) * np.sum(lineerFunction(x,w,b) - y)
    w -= a * dw
    b -= a * db
    return w,b