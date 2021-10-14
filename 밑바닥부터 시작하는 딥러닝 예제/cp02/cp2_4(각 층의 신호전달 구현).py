import numpy as np
from numpy.core.numeric import identity

def sigmoid(x): #시그모이드 함수
    return 1 / (1 + np.exp(-x))

def identity_function(x): #항등 함수
    return x

############# 입력층 -> 1층 #################

X = np.array([1.0, 0.5]) #입력층 입력값
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) #1층 가중치
B1 = np.array([0.1, 0.2, 0.3]) #1층 편향

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1 #a1 = w11x1 + w12x2 + b1 -> A1 = WX + B
Z1 = sigmoid(A1)

print(A1)
print(Z1)

############# 1층 -> 2층 #################

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) #2층 가중치
B2 = np.array([0.1, 0.2]) #2층 편향

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2 #a2 = w11z1 + w12z2 + w13z3 + b2 -> A2 = WX + B
Z2 = sigmoid(A2)

############# 2층 -> 출력층 #################

W3 = np.array([[0.1, 0.3], [0.2, 0.4]]) #출력층 가중치
B3 = np.array([0.1, 0.2]) #출력층 편향

A3 = np.dot(Z2, W3) + B3 # #a3 = w11z1 + w12z2 + b1 -> A3 = WX + B
Y = identity_function(A3) # Y = A3와 같음

print("Y : " + str(Y))