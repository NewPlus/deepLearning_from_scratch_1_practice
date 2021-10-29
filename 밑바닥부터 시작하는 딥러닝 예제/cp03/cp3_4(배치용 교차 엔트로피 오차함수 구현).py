import numpy as np

def cross_entropy_error(y, t): ## 원-핫 인코딩 용 교차 엔트로피 오차 for 배치
    if y.ndim == 1: ##신경망의 출력이 하나일 경우
        t = t.reshape(1, y.size) ## reshape으로 정답 레이블의 데이터의 형상 바꾸기
        y = y.reshape(1, y.size) ## reshape으로 신경망 출력 데이터의 형상 바꾸기
    
    batch_size = y.shape[0] 
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
    ## 배치 크기로 나눠 정규화 후 이미지 한장 당 평균의 교차엔트로피 오차를 구하고 그 값 리턴

def cross_entropy_error_for_number(y, t):
    if y.ndim == 1: ##신경망의 출력이 하나일 경우
        t = t.reshape(1, y.size) ## reshape으로 정답 레이블의 데이터의 형상 바꾸기
        y = y.reshape(1, y.size) ## reshape으로 신경망 출력 데이터의 형상 바꾸기
    
    batch_size = y.shape[0] 
    return -np.sum(np.log(y[np.arrange(batch_size), t] + 1e-7)) / batch_size
    ## np.arrage(batch_size)는 0부터 batch_size - 1까지 배열 생성(a라 칭함)
    ## t에는 이미 정답 레이블이 있음(b라 칭함)
    ## y에 [a,b]의 형태로 [y[a1,b1], y[a2,b2], y[a3,b3], ... , y[an,bn]] 생성
    ## (n은 batch_size와 같음)
    ## 그 다음은 엔트로피 오차를 구하는 과정임.