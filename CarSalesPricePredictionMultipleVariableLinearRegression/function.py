import copy
import math
import numpy as np


def linearFunction(x, w, b):
    return np.dot(x, w) + b

def costFunction(x, y, w, b, linearFunction):
    m = x.shape[0]
    J = 0.
    for i in range(m):
        J += (((linearFunction(x[i], w, b)) - y[i]) ** 2)
    J /= 2*m
    return J

def gradientFunction(x, y, w, b, linearFunction):
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = linearFunction(x[i], w, b) - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i][j]
        dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def run(x, y, w, b, linearFunction, costFunction, gradientFunction, alpha, iterations):
    hist_J = []
    w_final = copy.deepcopy(w)
    b_final = copy.deepcopy(b)

    for i in range(iterations):
        dj_dw, dj_db = gradientFunction(x, y, w_final, b_final, linearFunction)
        w_final -= alpha * dj_dw
        b_final -= alpha * dj_db

        hist_J.append(costFunction(x, y, w_final, b_final, linearFunction))

        if i % math.ceil(iterations/10) == 0:
            print(f"Iteration: {i}, Cost: {float(hist_J[-1]):.4f}")

    print(f"Final Weights: {w_final}")
    print(f"Final Bias: {b_final}")

    m = x.shape[0]
    for i in range(m):
        if i % math.ceil(m / 5) == 0:
            print(f"Prediction: {linearFunction(x[i], w_final, b_final)}, Target: {y[i]}")

    return w_final, b_final, hist_J