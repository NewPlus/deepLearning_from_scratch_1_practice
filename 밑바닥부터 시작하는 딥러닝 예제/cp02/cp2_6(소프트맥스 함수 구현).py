import numpy as np

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) #c는 최댓값을 빼주어 오버플로우 방지를 위함임
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

x = np.array([1010, 1000, 990])
print(softmax(x))