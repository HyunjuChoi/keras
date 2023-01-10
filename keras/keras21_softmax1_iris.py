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
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

print(y)
print(y.shape)                  #(150,3)

#one_hot_encode_y = pd.get_dummies(datasets[y])

#print(x, y)
#print(x.shape, y.shape)                 #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2,                        #shuffle = False로 설정 시 셔플 안돼서 맨 앞부터 차례로 비율 설정되어 같은 값만 뽑힘.
                                                                                #셔플 설정 안하면 예측값 좋지 않다!
    stratify= y                                                                 #<분류>에만 사용 가능! <회귀>모델이면 error 발생!                                                                 
)

#print(y_train)
#print(y_test)

### 데이터 양이 많아질수록 train과 test의 데이터 값이 한쪽으로 치우치게 분류되어 예측 모델 성능 하락할 수 있음.###
### => 따라서 분류할 때 한쪽 데이터에만 치우치지 않게 해주는 것! *** stratify= y ***   데이터 개수 동일한 비율로 뽑힌다!


#2. modeling
model = Sequential()
model.add(Dense(40, activation='relu', input_shape=(4, )))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(3, activation='softmax'))                       #다중 분류!!!   =>y 데이터 종류 개수(class) 0, 1, 2 세개이므로 output layer도 3이다

#3.compile and training
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.2, verbose=1)

#4. evaluation prediction
loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy:', accuracy)


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
y_predict = model.predict(x_test)
#print(y_predict)
'''
[[9.99776542e-01 2.23436815e-04 9.26398019e-21]
 [1.05961290e-11 2.23073922e-03 9.97769237e-01]
 [9.99832511e-01 1.67454782e-04 4.95978020e-21]
 [4.50488313e-09 1.30853906e-01 8.69146109e-01]
 [7.26735028e-09 1.76931351e-01 8.23068559e-01]
 [1.53647221e-07 7.49591768e-01 2.50408053e-01]
 [9.99768794e-01 2.31153754e-04 1.31838574e-20]
 [1.43495025e-11 2.98978994e-03 9.97010231e-01]
 [9.99783576e-01 2.16440225e-04 9.45296269e-21]
 [1.48966135e-11 3.00275115e-03 9.96997237e-01]
 [4.89722585e-12 1.46913237e-03 9.98530865e-01]
 [3.06293574e-10 2.42768023e-02 9.75723147e-01]
 [4.18809071e-10 2.84040906e-02 9.71595943e-01]
 [9.99821723e-01 1.78356175e-04 5.38971217e-21]
 [9.99790609e-01 2.09458885e-04 9.07889853e-21]
 [9.99806345e-01 1.93703498e-04 6.72702775e-21]
 [5.20687356e-12 1.40721910e-03 9.98592794e-01]
 [9.99891639e-01 1.08364271e-04 1.40637850e-21]
 [3.01313002e-10 2.28192266e-02 9.77180719e-01]
 [4.48502442e-06 9.99267161e-01 7.28269457e-04]
 [9.99754608e-01 2.45428440e-04 1.25351087e-20]
 [7.47441109e-10 4.02933136e-02 9.59706664e-01]
 [2.24269973e-03 9.97757375e-01 1.83430942e-08]
 [1.57563170e-06 9.94338691e-01 5.65972226e-03]
 [9.99808013e-01 1.91956438e-04 7.62750432e-21]
 [2.52046908e-11 4.01195884e-03 9.95988071e-01]
 [1.20073253e-06 9.87487674e-01 1.25111751e-02]
 [1.82416788e-05 9.99926329e-01 5.54396247e-05]
 [1.64315843e-05 9.99884367e-01 9.92263012e-05]
 [2.84527846e-09 8.83064121e-02 9.11693633e-01]]
'''
y_predict = np.argmax(y_predict, axis=1)                        #argmax: y_predict의 가장 큰 값 => 자리값 뽑아줌
#print(y_test)
'''
[[1. 0. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]]
'''

y_test = np.argmax(y_test, axis=1)
# print('y_pred 예측값: ',y_predict)                                                   #[0 2 0 2 2 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 2]
# print('y_test 원래 값: ',y_test)                                                       #[0 2 0 1 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]
acc = accuracy_score(y_test, y_predict)                                           #y_test: 정수형, y_predict는 실수형이라 error 남

print('acc: ', acc)