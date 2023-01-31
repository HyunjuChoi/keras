from sklearn.metrics import accuracy_score
import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM, Conv1D
from sklearn.model_selection import train_test_split
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

path = 'C:/study/_save/'

#1. data
datasets = load_digits()
x = datasets['data']
y = datasets['target']
print(x.shape, y.shape)                             # (1797, 64) (1797,)
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),               =>다중분류 data
print(np.unique(y, return_counts=True))
#  array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[3])
# plt.show()

y = to_categorical(y)

# print(y)
# print(y.shape)                  #(1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=115,
                                                    test_size=0.2, stratify=y)

# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, x_test.shape)                  # (1437, 64) (360, 64)

print(y_test.shape)         # (360, 10)

x_train = x_train.reshape(1437, 8, 8)
x_test = x_test.reshape(360, 8, 8)

#2.modeling
model = Sequential()
model.add(Conv1D(30, 2, activation='linear', input_shape=(8, 8)))
model.add(Conv1D(20, 2, activation='relu'))
model.add(Conv1D(50, 2, activation='relu'))
model.add(Conv1D(30, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(20, activation='linear'))
model.add(Dense(10, activation='softmax'))

#2.modeling
# input1 = Input(shape=(64, ))
# dense1 = Dense(30, activation='linear')(input1)
# drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(20, activation='relu')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(50, activation='relu')(drop2)
# drop3 = Dropout(0.2)(dense3)
# dense4 = Dense(30, activation='relu')(drop3)
# dense5 = Dense(20, activation='linear')(dense4)
# output1 = Dense(10, activation='softmax')(dense5)

# model = Model(inputs = input1, outputs = output1)

#3. compile and training

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    #baseline='0.1',                특정 값 도달 시 훈련 중지
    restore_best_weights=True,
    verbose=1
)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

date = datetime.datetime.now()
print(date)
print(type(date))                           # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")

print(date)                                 # 0112_1502

filepath = 'C:/study/_save/MCP/'
# d: digit, f: float
filename = '{epoch:04d}-{val_loss: .4f}.hdf5'


#ModelCheckpoint 설정
# mcp = ModelCheckpoint(
#     monitor='val_loss', mode='auto', verbose=1,
#     save_best_only=True,                                              # save_best_only: 가중치 가장 좋은 지점 저장!
#     # filepath= path + 'MCP/keras30_ModelCheckPoint3.hdf5'
#     filepath= filepath + 'k51_digits_' + 'd_'+ date + '_'+ 'e_v_'+ filename                      #파일명 날짜, 시간 넣어서 저장하기
# )

model.fit(x_train, y_train, epochs=1000, batch_size=5,
          validation_split=0.2, verbose=1, callbacks=[es])

# model.save(path + 'keras51_conv1d_save_model_digits.h5')

#4. evaluation and prediction
loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy: ', accuracy)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)

y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test, y_predict)

print('acc: ', acc)


'''
1/11일

<<min max>>
Epoch 00064: early stopping
12/12 [==============================] - 0s 1ms/step - loss: 0.2540 - accuracy: 0.9444
loss:  0.25403422117233276
accuracy:  0.9444444179534912
acc:  0.9444444444444444

<<standard>>
Epoch 00060: early stopping
12/12 [==============================] - 0s 635us/step - loss: 0.2434 - accuracy: 0.9500
loss:  0.2433536946773529
accuracy:  0.949999988079071
acc:  0.95



1/25
<< cnn >> 
Epoch 00059: early stopping
12/12 [==============================] - 0s 1ms/step - loss: 0.2438 - accuracy: 0.9583
loss:  0.24375073611736298
accuracy:  0.9583333134651184
acc:  0.9583333333333334


1/27 << LSTM >>
 Epoch 00078: early stopping
12/12 [==============================] - 0s 1ms/step - loss: 0.3097 - accuracy: 0.9306
loss:  0.30965831875801086
accuracy:  0.9305555820465088
acc:  0.9305555555555556

1/28 << conv 1d>>
Epoch 00062: early stopping
12/12 [==============================] - 0s 2ms/step - loss: 0.1630 - accuracy: 0.9583
loss:  0.16297541558742523
accuracy:  0.9583333134651184
acc:  0.9583333333333334
'''