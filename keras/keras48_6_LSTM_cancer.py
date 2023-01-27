from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint                    #EarlyStopping 추가

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


path2= 'C:/study/_save/'


#1. data
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets['data']
y= datasets['target']
# #print(x.shape, y.shape)         #(569, 30) (569,)

x_train, x_test, y_train, y_test= train_test_split(x, y, shuffle=True, random_state=333, test_size=0.2)

# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, x_test.shape)                  # (455, 30) (114, 30)

x_train = x_train.reshape(455, 6, 5)
x_test = x_test.reshape(114, 6, 5)

#2. modeling
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(6,5)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32,activation='relu',))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))                           #이진분류 activation = 'sigmoid'로 고정!!


#2. modeling
# input1 = Input(shape = (30, ))
# dense1 = Dense(50, activation='linear')(input1)
# drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(40, activation='relu')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(30, activation='relu')(drop2)
# drop3 = Dropout(0.2)(dense3)
# dense4 = Dense(20, activation='relu')(drop3)
# dense5 = Dense(10, activation='relu')(dense4)
# output1 = Dense(1, activation='sigmoid')(dense5)

# model = Model(inputs = input1, outputs = output1)

#3. compile and training
model.compile(loss='binary_crossentropy', optimizer='adam',         #이진분류 무조건 loss='binary_crossentropy'
                metrics=['accuracy'])                               #정확도 판단 가능
                                                                    #metrics 추가 시 hist의 history에도 accuracy 정보 추가 됨


#earlystopping 기준 설정
es = EarlyStopping(
    monitor='val_loss',                     #history의 val_loss의 최소값을 이용함
    mode='min',                             #max로 설정 시 갱신 안됨. (accuracy는 높을수록 좋기 때문에 max로 설정)
    patience=20,                            #earlystopping 5번 (최저점 나올 때까지 5번 돌림. 그 안에 안 나오면 종료) 
    restore_best_weights=True,
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
#     filepath= filepath + 'k48_cancer_' + 'd_'+ date + '_'+ 'e_v_'+ filename                      #파일명 날짜, 시간 넣어서 저장하기            
# )


hist = model.fit(x_train, y_train, epochs=5000, batch_size=16, validation_split=0.2,
          callbacks=[es], verbose=1)

# model.save(path2 + 'keras48_LSTM_save_model_cancer.h5')   
      
#4. evaluation and prediction

# loss= model.evaluate(x_test, y_test)
# print('loss, accuracy: ', loss)

loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ',loss)
print('accuracy: ', accuracy)

# results = model.predict()

y_predict = model.predict(x_test)                   #sigmoid 통과 후 값

# print(y_predict[:10])
# print(y_test[:10])

from sklearn.metrics import r2_score, accuracy_score                    #Accuracy Score 추가

# print(y_predict)
# print(y_test)

#acc = accuracy_score(y_test, y_predict)                               #그냥 돌리면 y_test랑 y_predict 실수/정수형이라 에러남


# 구글링해서 찾은 방법
# y_pred_1d = y_predict.flatten()                                         # 차원 펴주기, numpy에서 제공하는 다차원 배열 공간을 1차원으로 평탄화해주는 함수
# y_pred_class = np.where(y_pred_1d > 0.5, 1 , 0)                         #조건을 찾아, 변경하거나, 인덱싱하는 간단한 함수가 numpy의 where함수(조건문)
                                                                          # 0.5보다 크면 1, 작으면 0


#팀원 구글링
#y_predict = np.asarray(y_predict, dtype=int)                             #얘는 왜 다 0으로 바뀔까?

#다른 방법
y_int = np.round(y_predict).astype(int)
# print(y_int)


# print(y_pred_1d)                           왜 여기서 다 0.5보다 크게 나오는데
# print(y_pred_class)                       여기서 어쩔 땐 1이고 어쩔 땐 0이지????

acc = accuracy_score(y_test, y_int)
print('accuracy score: ', acc)
'''
print(y_predict)
print(y_test)
print(y_pred_class)
print('=================================')
print(hist.history)



1/11일

<<min max>>
Epoch 00064: early stopping
4/4 [==============================] - 0s 745us/step - loss: 0.1875 - accuracy: 0.9561
loss:  0.18745160102844238
accuracy:  0.9561403393745422
accuracy score:  0.956140350877193

<<standard>>
Epoch 00045: early stopping
4/4 [==============================] - 0s 984us/step - loss: 0.2211 - accuracy: 0.9561
loss:  0.22107712924480438
accuracy:  0.9561403393745422
accuracy score:  0.956140350877193



1/12

<<dropout, mcp 적용>>
Epoch 00050: early stopping
4/4 [==============================] - 0s 896us/step - loss: 0.1253 - accuracy: 0.9825
loss:  0.1253311038017273
accuracy:  0.9824561476707458
accuracy score:  0.9824561403508771


1/25

<<cnn>>
Epoch 00029: early stopping
4/4 [==============================] - 0s 2ms/step - loss: 0.1148 - accuracy: 0.9737
loss:  0.11477396637201309
accuracy:  0.9736841917037964
accuracy score:  0.9736842105263158


1/27 << LSTM >>
Epoch 00062: early stopping
4/4 [==============================] - 0s 2ms/step - loss: 0.3899 - accuracy: 0.9298
loss:  0.3899337947368622
accuracy:  0.9298245906829834
accuracy score:  0.9298245614035088

'''