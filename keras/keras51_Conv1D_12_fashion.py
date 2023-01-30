import datetime
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, AveragePooling2D, Dropout, LSTM, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# 텐서플로 mnist 데이터셋 불러와서 변수에 저장하기
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

# print(x_train, y_train)
# (60000, 28, 28) (60000,)=>(흑백데이터)
print(x_train.shape, y_train.shape)
# (10000, 28, 28) (10000,)
print(x_test.shape, y_test.shape)


print(x_train[0])
print(y_train[0])
'''
plt.imshow(x_train[115], 'Blues')
plt.show
'''

#2. 모델구성

model = Sequential()
model.add(Conv1D(128, 2, activation='relu', input_shape=(28, 28)))
model.add(Conv1D(128, 2, activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(64, 2, activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(32, 2, activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(32, 2, activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(32, 2, activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))  # output 노드 10개이므로 다중분류!

#model.summary()


#3. 컴파일 & 훈련

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
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
#     filepath= filepath + 'k51_conv1d_fashion_' + 'd_'+ date + '_'+ 'e_v_'+ filename
# )

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100, verbose=3,
          batch_size=128, validation_split=0.2, callbacks=[es])

#4. 평가 & 예측
results = model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])
print("Accuracy: %.2f%%" % (results[1]*100))


'''결과치

1. << max pooling, average pooling, padding 추가>>

Epoch 00036: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.3052 - acc: 0.9021
loss:  0.30520984530448914
acc:  0.9021000266075134
Accuracy: 90.21%

2.  conv2D 레이어 여러개 추가, dropout 추가
313/313 [==============================] - 1s 3ms/step - loss: 0.2687 - acc: 0.9019
loss:  0.26868292689323425
acc:  0.9018999934196472
Accuracy: 90.19%


<< dnn 방식 >>
1. 
loss:  0.7739614844322205
acc:  0.7006000280380249
Accuracy: 70.06%

2. early stopping = false
313/313 [==============================] - 0s 793us/step - loss: 0.9156 - acc: 0.6203
loss:  0.9155704975128174
acc:  0.6202999949455261
Accuracy: 62.03%

3. early = true, 히든레이어 추가
313/313 [==============================] - 0s 722us/step - loss: 1.0357 - acc: 0.6178
loss:  1.0356634855270386
acc:  0.6177999973297119
Accuracy: 61.78%



1/27 << LSTM >>
Epoch 00042: early stopping
313/313 [==============================] - 2s 6ms/step - loss: 2.3026 - acc: 0.1000
loss:  2.3025918006896973
acc:  0.10000000149011612
Accuracy: 10.00%
'''