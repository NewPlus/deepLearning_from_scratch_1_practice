import sys, os

from numpy.lib import twodim_base
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) =\
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10 ## 배치할 사이즈
batch_mask = np.random.choice(train_size, batch_size)
## numpy의 랜덤함수로 0 <= x < 60000의 범위에서 무작위로 batch_size 만큼의 수를 뽑아냄
x_batch = x_train[batch_mask] ## 그 뽑아낸 수를 데이터의 인덱스로 사용
t_batch = t_train[batch_mask]

print(x_batch.shape)
print(t_batch.shape)