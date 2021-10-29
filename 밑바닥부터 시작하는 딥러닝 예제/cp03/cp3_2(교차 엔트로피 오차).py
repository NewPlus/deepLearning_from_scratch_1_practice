import numpy as np

def cross_entropy_error(y, t): ##교차 엔트로피 오차 함수
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] ## 정답은 '2'
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
## 정답을 2로 예측한 경우(0.6)
print(cross_entropy_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
## 정답을 7로 예측한 경우(0.6)
print(cross_entropy_error(np.array(y), np.array(t)))