import datetime
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, AveragePooling2D, LSTM, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10, cifar100  # 컬러 이미지 데이터
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
# 텐서플로 mnist 데이터셋 불러와서 변수에 저장하기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train, y_train)
# (50000, 32, 32, 3) (50000, 1)
print(x_train.shape, y_train.shape)
# (10000, 32, 32, 3) (10000, 1)
print(x_test.shape, y_test.shape)

# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
print(np.unique(y_train, return_counts=True))
# array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))


x_train = x_train.reshape(50000, 32*3, 32)
x_test = x_test.reshape(10000, 32*3, 32)

x_train = x_train/255.  # 이거 뭐야?
x_test = x_test/255.


#2. 모델구성

model = Sequential()
model.add(Conv1D(128, 2, activation='relu', input_shape=(32*3, 32)))
model.add(Conv1D(64, 2, activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(32, 2, activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(32, 2, activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(16, 2, activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 & 훈련

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=False,
    verbose=3
)

date = datetime.datetime.now()
print(date)
print(type(date))                           # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")

print(date)

# filepath = 'C:/study/_save/MCP/'
filepath = '/Users/hyunju/Desktop/study/_save/MCP/'

# d: digit, f: float
filename = '{epoch:04d}-{val_loss: .4f}.hdf5'


# mcp = ModelCheckpoint(
#     monitor='val_loss', mode='auto', verbose=3,
#     save_best_only=True,
#     filepath= filepath + 'k51_conv1d_cifar10_' + 'd_'+ date + '_'+ 'e_v_'+ filename
# )

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100,  batch_size=100,
          callbacks=[es])

#4. 평가 & 예측
results = model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])

'''
1/13 결과치
<<es, mcp 적용>>

313/313 [==============================] - 1s 3ms/step - loss: 2.3026 - acc: 0.1000
loss:  2.3026461601257324
acc:  0.10000000149011612

2. <<kernel_size, filters 값 변경 및 conv레이어 추가, batch_size= 128>>
loss:  3.044217109680176
acc:  0.48080000281333923

3. epoch= 10 , batch = 100
loss:  1.9259449243545532
acc:  0.4878000020980835

4.
loss:  1.842490792274475
acc:  0.4966000020503998

5.
Epoch 00026: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 3.0226 - acc: 0.4695
loss:  3.022554636001587
acc:  0.46950000524520874



1/25

1.  drop out, maxpooling, average pooling, padding 추가

313/313 [==============================] - 2s 4ms/step - loss: 0.9347 - acc: 0.6792
loss:  0.9346588850021362
acc:  0.6791999936103821


<<dnn>>

1.
Epoch 00025: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 2.1926 - acc: 0.1393
loss:  2.1925742626190186
acc:  0.13930000364780426


1/27 << LSTM >>

1. epochs= 1
500/500 [==============================] - 36s 70ms/step - loss: 2.4795 - acc: 0.1069
313/313 [==============================] - 4s 14ms/step - loss: 2.2955 - acc: 0.1433
loss:  2.295531749725342
acc:  0.14329999685287476

2. epochs=100
313/313 [==============================] - 11s 36ms/step - loss: 2.1591 - acc: 0.1841
loss:  2.1591241359710693
acc:  0.18410000205039978s
'''