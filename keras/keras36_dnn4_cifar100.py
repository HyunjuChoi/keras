from tensorflow.keras.datasets import cifar10, cifar100                 #컬러 이미지 데이터
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()                  #텐서플로 mnist 데이터셋 불러와서 변수에 저장하기

# print(x_train, y_train)
print(x_train.shape, y_train.shape)                     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)                       # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))          


x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3) 
                                                    
#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(32*32*3, )))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='softmax'))

#3. 컴파일 & 훈련

es = EarlyStopping(
     monitor='val_loss',
     mode='min',
     patience=20,
     restore_best_weights=True,           
     verbose=3
)

import datetime
date = datetime.datetime.now()
print(date)
print(type(date))                           # <class 'datetime.datetime'>
date= date.strftime("%m%d_%H%M")

print(date)                                 

filepath = 'C:/study/_save/MCP/'
filename = '{epoch:04d}-{val_loss: .4f}.hdf5'                       # d: digit, f: float 


mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=3,
    save_best_only=True,
    filepath= filepath + 'k36_dnn_cifar100_' + 'd_'+ date + '_'+ 'e_v_'+ filename 
)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100, verbose=3, batch_size=128, validation_split=0.2,
          callbacks=[es, mcp])

#4. 평가 & 예측
results = model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])



'''
1/13 결과치
<<es, mcp 적용>>
Epoch 00021: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 4.6052 - acc: 0.0100 
loss:  4.605203628540039
acc:  0.009999999776482582

Epoch 00021: early stopping
313/313 [==============================] - 1s 4ms/step - loss: 4.6052 - acc: 0.0100 
loss:  4.60520601272583
acc:  0.009999999776482582



1/25
1. drop out maxpooling averagepooling padding conv2D 레이어 추가
313/313 [==============================] - 2s 5ms/step - loss: 2.7627 - acc: 0.2888
loss:  2.762747287750244
acc:  0.2888000011444092


2. <drop out 추가>
313/313 [==============================] - 1s 4ms/step - loss: 3.2827 - acc: 0.1778
loss:  3.282677412033081
acc:  0.1777999997138977


3. <2의 dropout 제거, 다른 drop out 비율 변경>



<< dnn 방식>>
1.
Epoch 00033: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 4.5030 - acc: 0.0208
loss:  4.503002166748047
acc:  0.020800000056624413

2.
313/313 [==============================] - 1s 3ms/step - loss: nan - acc: 0.0100
loss:  nan
acc:  0.009999999776482582
'''