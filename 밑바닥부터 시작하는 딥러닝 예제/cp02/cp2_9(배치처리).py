import pickle
import sys, os
sys.path.append(os.pardir) #부모 디렉토리 불러오기용
import numpy as np
import pickle
from dataset.mnist import load_mnist

def get_data(): #MNIST 데이터를 반환하는 함수
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=False, one_hot_label=False) #(훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블) 형식
        #normalize를 통해 픽셀 값을 0 ~ 1사이로 정규화(Fasle시, 0 ~ 255 사이값)
        #flatten을 통해 1차원으로 만들지(여기서는 1*28*28의 3차원임)
        #one_hot_label을 통해 숫자별 리스트로 출력받을지(여기서는 그냥 숫자 형태 레이블로 저장)
    return x_test, t_test

def init_network(): #파일에 저장된 '학습된 가중치 매개변수'를 읽음
    with open("C:/Users/lyhth/OneDrive/Desktop/Programming/git/vscode-git/밑바닥부터 시작하는 딥러닝 예제/cp02/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    
    return network

def sigmoid(x): #시그모이드 함수
    return 1 / (1 + np.exp(-x))

def softmax(a): #소프트맥스 함수
    c = np.max(a)
    exp_a = np.exp(a - c) #c는 최댓값을 빼주어 오버플로우 방지를 위함임
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

############## 배치 처리 추가 #################
batch_size = 100 # 배치 크기
##############################################

accuracy_cnt = 0

############## 배치 처리 추가 #################
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1) #확률이 가장 높은 원소의 인덱스를 얻는다(그게 신경망이 생각하는 정답값)
    # axis=1이므로 1번째 차원이 축임
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
##############################################

print("Accuracy : " + str(float(accuracy_cnt) / len(x)))