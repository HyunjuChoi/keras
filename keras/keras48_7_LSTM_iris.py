from sklearn.datasets import load_iris                      #꽃잎 정보 가지고 무슨 꽃인지 맞추기
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

path= 'C:/study/_save/'

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

# print(y)
# print(y.shape)                  #(150,3)

#print(x, y)
#print(x.shape, y.shape)                #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2,                        #shuffle = False로 설정 시 셔플 안돼서 맨 앞부터 차례로 비율 설정되어 같은 값만 뽑힘.
                                                                                #셔플 설정 안하면 예측값 좋지 않다!
    stratify= y                                                                 #<분류>에만 사용 가능! <회귀>모델이면 error 발생!                                                                 
)

# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, x_test.shape)                      # (120, 4) (30, 4)

x_train = x_train.reshape(120, 2, 2)
x_test = x_test.reshape(30, 2, 2)

### 데이터 양이 많아질수록 train과 test의 데이터 값이 한쪽으로 치우치게 분류되어 예측 모델 성능 하락할 수 있음.###
### => 따라서 분류할 때 한쪽 데이터에만 치우치지 않게 해주는 것! *** stratify= y ***   데이터 종류 동일한 비율로 뽑힌다!


#2. modeling
model = Sequential()
model.add(LSTM(40, activation='relu', input_shape=(2, 2)))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(3, activation='softmax'))                       #다중 분류!!!   =>y 데이터 종류 개수(class) 0, 1, 2 세개이므로 output layer도 3이다


#2. modeling
# input1 = Input(shape= (4, ))
# dense1 = Dense(40, activation='relu')(input1)
# drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(30, activation='sigmoid')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(20, activation='relu')(drop2)
# drop3 = Dropout(0.2)(dense3)
# dense4 = Dense(10, activation='linear')(drop3)
# output1 = Dense(3, activation='softmax')(dense4)

# model = Model(inputs = input1, outputs = output1)

#3.compile and training
# model.compile(loss='categorical_crossentropy', optimizer='adam',
#               metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam',                     
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint                    #ModelCheckpoint 추가

#earlystopping 기준 설정
es = EarlyStopping(
    monitor='val_loss',                     # history의 val_loss의 최소값을 이용함
    mode='min',                             # max로 설정 시 갱신 안됨. (accuracy 사용 시에는 정확도 높을수록 좋기 때문에 max로 설정)
    patience=10,                            # earlystopping n번 (최저점 나올 때까지 n번 돌림. 그 안에 안 나오면 종료) 
    restore_best_weights=False,              # 이걸 설정해줘야 종료 시점이 아닌 early stopping 지점의 최적 weight 값 사용 가능
    verbose=1
)

import datetime
date = datetime.datetime.now()
print(date)
print(type(date))                           # <class 'datetime.datetime'>
date= date.strftime("%m%d_%H%M")

print(date)                                 # 0112_1502

filepath = 'C:/study/_save/MCP/'
filename = '{epoch:04d}-{val_loss: .4f}.hdf5'                       # d: digit, f: float 


#ModelCheckpoint 설정
# mcp = ModelCheckpoint(
#     monitor='val_loss', mode='auto', verbose=1,
#     save_best_only=True,                                              # save_best_only: 가중치 가장 좋은 지점 저장!
#     # filepath= path + 'MCP/keras30_ModelCheckPoint3.hdf5' 
#     filepath= filepath + 'k48_iris_' + 'd_'+ date + '_'+ 'e_v_'+ filename                      #파일명 날짜, 시간 넣어서 저장하기            
# )

model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.2, verbose=1, callbacks=[es])

# model.save(path + 'keras48_LSTM_save_model_iris.h5')    
  
#4. evaluation prediction
loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy:', accuracy)


# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print('y_predict: ',y_predict)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
#print(y_predict)

y_predict = np.argmax(y_predict, axis=1)                        #argmax: y_predict의 가장 큰 값 => 자리값 뽑아줌
#print(y_test)

y_test = np.argmax(y_test, axis=1)
# print('y_pred 예측값: ',y_predict)                                                   #[0 2 0 2 2 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 2]
# print('y_test 원래 값: ',y_test)                                                       #[0 2 0 1 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]
acc = accuracy_score(y_test, y_predict)                                           #y_test: 정수형, y_predict는 실수형이라 error 남

print('acc: ', acc)


'''
1/11일

<<min max>>
loss:  0.15744414925575256
accuracy: 0.8999999761581421
acc:  0.9

<<standard>>
loss:  1.0140268802642822
accuracy: 0.8666666746139526
acc:  0.8666666666666667


1/12
<<dropout, mcp>>
Epoch 00042: early stopping
1/1 [==============================] - 0s 82ms/step - loss: 0.2741 - accuracy: 0.9000
loss:  0.2741049528121948
accuracy: 0.8999999761581421
acc:  0.9




1/25
<< cnn >>
Epoch 00038: early stopping
1/1 [==============================] - 0s 87ms/step - loss: 0.2872 - accuracy: 0.9000
loss:  0.28723496198654175
accuracy: 0.8999999761581421
acc:  0.9


1/27 << LSTM >>
Epoch 00031: early stopping
1/1 [==============================] - 0s 166ms/step - loss: 0.3239 - accuracy: 0.9000
loss:  0.3238670229911804
accuracy: 0.8999999761581421
acc:  0.9


'''