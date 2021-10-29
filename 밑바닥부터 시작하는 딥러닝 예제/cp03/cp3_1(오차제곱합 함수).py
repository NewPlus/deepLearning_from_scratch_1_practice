import numpy as np

def sum_squares_error(y,t): ## 오차제곱합 함수
    return 0.5 * np.sum((y-t)**2)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] ## 정답은 2

y = [0.1, 0.005, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
## 예1 : '2'일 확률이 가장 높다고 추정한 경우(0.6)
print(sum_squares_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
## 예1 : '7'일 확률이 가장 높다고 추정한 경우(0.6)
print(sum_squares_error(np.array(y), np.array(t)))