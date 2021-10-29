import numpy as np

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

## x0에 대한 편미분을 구할때 (단, x0 = 3, x1 = 4)
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

print(numerical_diff(function_tmp1, 3.0))

## x1에 대한 편미분을 구할때 (단, x0 = 3, x1 = 4)
def function_tmp2(x1):
    return x1*x1 + 3.0**2.0

print(numerical_diff(function_tmp2, 4.0))