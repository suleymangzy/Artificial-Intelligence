import copy
import math
import numpy as np


def linearFunction(x, w, b):
    return w * x + b

def costFunction(x, y, w, b, linear_function):
    m = x.shape[0]
    f = linear_function(x, w, b)
    error = f - y
    return (1 / (2 * m)) * np.sum(np.square(error))

def gradientFunction(x, y, w, b, linear_function):
    m = x.shape[0]
    dj_dw  = (1/m) * np.dot((linear_function(x, w, b) - y), x)
    dj_db = (1/m) * np.sum(linear_function(x, w, b) - y)
    return dj_dw, dj_db

def run(x, y, w, b, linear_function, cost_function, gradient_function, alpha, iterations):
    hist_J = []
    w_final = copy.deepcopy(w)
    b_final = copy.deepcopy(b)

    for i in range(iterations):
        dj_dw, dj_db = gradient_function(x, y, w_final, b_final, linear_function)
        w_final -= alpha * dj_dw
        b_final -= alpha * dj_db

        cost = cost_function(x, y, w_final, b_final, linear_function)
        hist_J.append(cost)

        if i % math.ceil(iterations/10) == 0:
            print(f"Iteration: {i}, Cost: {cost}")

    print(f"Weight: {w_final}, Bias: {b_final}")
    m = x.shape[0]
    for i in range(m):
        if i % math.ceil(m/5) == 0:
            print(f"Prediction: {linearFunction(x[i],w_final,b_final)}, Target: {y[i]}")

    return w_final, b_final, hist_J

