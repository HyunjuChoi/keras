from sklearn.datasets import load_iris                      #꽃잎 정보 가지고 무슨 꽃인지 맞추기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd

#1. data
datasets = load_iris()
#print(datasets.DESCR)                           #pandas=> .describe()  / .info()
'''
x_columns = 4, y_columns = 1
  ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================
'''
# print(datasets.feature_names)                  #pandas => .columns
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x  = datasets.data
y = datasets['target']


# one hot encoding 방법1
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

# print(y)
# print(y.shape)                  #(150,3)

#print(x, y)
#print(x.shape, y.shape)                #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2,                        #shuffle = False로 설정 시 셔플 안돼서 맨 앞부터 차례로 비율 설정되어 같은 값만 뽑힘.
                                                                                #셔플 설정 안하면 예측값 좋지 않다!
    stratify= y                                                                 #<분류>에만 사용 가능! <회귀>모델이면 error 발생!                                                                 
)
### 데이터 양이 많아질수록 train과 test의 데이터 값이 한쪽으로 치우치게 분류되어 예측 모델 성능 하락할 수 있음.###
### => 따라서 분류할 때 한쪽 데이터에만 치우치지 않게 해주는 것! *** stratify= y ***   데이터 종류 동일한 비율로 뽑힌다!


#2. modeling
model = Sequential()
model.add(Dense(40, activation='relu', input_shape=(4, )))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(3, activation='softmax'))                       #one hot은 하지 않아도 y 노드 최종 개수만큼 output layer 3으로 설정해줘야 함.

#3.compile and training
# model.compile(loss='categorical_crossentropy', optimizer='adam',
#               metrics=['accuracy'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',                     #one hot encoding 안 할 경우 loss를 sparse_categorical_crossentropy로 변경하면 됨
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.2, verbose=1)

#4. evaluation prediction
loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy:', accuracy)

\
# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print('y_predict: ',y_predict)
'''
y_predict:  [[9.9995244e-01 4.7606685e-05 3.4614822e-16]
 [4.9353897e-08 4.7264332e-03 9.9527353e-01]
 [9.9996567e-01 3.4294448e-05 2.0450516e-16]
 [5.8907603e-06 4.3089655e-01 5.6909758e-01]
 [8.3191026e-06 5.4580420e-01 4.5418742e-01]
 [1.5800082e-05 9.2398739e-01 7.5996868e-02]
 [9.9994445e-01 5.5564858e-05 4.3512368e-16]
 [5.1671343e-08 5.0759800e-03 9.9492395e-01]
 [9.9996018e-01 3.9839557e-05 3.2733178e-16]
 [6.9360460e-08 6.5446561e-03 9.9345523e-01]]
'''

from sklearn.metrics import accuracy_score

#x_test.shape = (30,4)

y_predict = model.predict(x_test)                               #y_predict.shape = (30,3)
y_predict = np.argmax(y_predict, axis=1)                        #argmax: y_predict의 가장 큰 값 => 자리값 뽑아줌
# print(y_test)

# y_test = np.argmax(y_test, axis=1)                            #one hot 안했으므로 argmax 할 필요 X
# print('y_pred 예측값: ',y_predict)                                                   #[0 2 0 2 2 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 2]
# print('y_test 원래 값: ',y_test)                                                       #[0 2 0 1 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]
acc = accuracy_score(y_test, y_predict)                                           #y_test: 정수형, y_predict는 실수형이라 error 남

print('acc: ', acc)