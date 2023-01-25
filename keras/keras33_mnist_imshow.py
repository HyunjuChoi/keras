import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()                  #텐서플로 mnist 데이터셋 불러와서 변수에 저장하기

# print(x_train, y_train)
print(x_train.shape, y_train.shape)                     # (60000, 28, 28) (60000,)=>(흑백데이터)
print(x_test.shape, y_test.shape)                       # (10000, 28, 28) (10000,)


print(x_train[100])
print(y_train[100])

plt.imshow(x_train[115], 'Blues')
plt.show