from tensorflow.keras.datasets import cifar10, cifar100                 #컬러 이미지 데이터
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()                  #텐서플로 mnist 데이터셋 불러와서 변수에 저장하기

# print(x_train, y_train)
print(x_train.shape, y_train.shape)                     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)                       # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))          

"""
(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
       68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
       85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), 
       array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64))

"""

                                                     
#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Dropout

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(32, 32, 3),
                 activation='relu'))                    # (31, 31, 128)
model.add(Conv2D(filters=64, kernel_size=(2,2)))        # (30, 30, 64)
model.add(Conv2D(filters=32, kernel_size=(2,2)))        # (28, 28, 32)
model.add(Flatten())                                    # 25,088
model.add(Dense(32, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='softmax'))

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
    filepath= filepath + 'k34_cifar100_' + 'd_'+ date + '_'+ 'e_v_'+ filename 
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
'''