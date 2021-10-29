import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_1_diff(f, x):
    diff = numerical_diff(f, x) ## 미분 계수
    y = f(x) - diff*x
    return lambda t: diff*t + y

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

tf = function_1_diff(function_1, 5)
y_diff = tf(x)

plt.xlabel("x")
plt.ylabel("f(x)")

plt.plot(x, y)
plt.plot(x, y_diff)
plt.show()