
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten,LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pylab as plt

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

path = 'C:/study/_save/'        


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

import time

#1. data
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

print(x.shape)          #(20640, 8)
print(y)
print(y.shape)          #(20460, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=115)

#scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, x_test.shape)                  # (14447, 8) (6193, 8)

x_train = x_train.reshape(14447, 4, 2)
x_test = x_test.reshape(6193, 4, 2)


# #2. modeling
model = Sequential()
model.add(Conv1D(128, 2, input_shape=(4, 2), activation='relu', padding='same'))
model.add(Conv1D(64, 2, padding='same'))
model.add(Conv1D(64, 2, padding='same'))
model.add(Conv1D(32, 2, padding='same'))
model.add(Conv1D(32, 2, padding='same'))
model.add(Conv1D(16, 2, padding='same'))
model.add(Conv1D(16, 2, padding='same'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

#3. compile and training
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

es = EarlyStopping(
    monitor='loss',
    mode = min,
    patience=10,
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
#     filepath= filepath + 'k51_cali_' + 'd_'+ date + '_'+ 'e_v_'+ filename                      #파일명 날짜, 시간 넣어서 저장하기            
# )

model.fit(x_train, y_train, epochs=1000, batch_size=500, validation_split=0.2, verbose=1,
                callbacks=[es])

# model.save(path + 'keras51_conv1d_save_model_california.h5')   

#4. evaluation prediction
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
print('r2: ',r2)

'''
#5.시각화
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('california loss')
plt.legend()
plt.show()
'''



'''결과치
1.  epochs=1000, batch_size=500,
    patience=10,
    
Epoch 00104: early stopping
194/194 [==============================] - 0s 1ms/step - loss: 1.3602 - mse: 1.3602
r2:  -0.012046967812149756
    
    
2. 
Epoch 00033: early stopping
194/194 [==============================] - 0s 1ms/step - loss: 1.3140 - mse: 1.3140
r2:  0.02228096660658907



1/11일

<<min-max scaling>>
Epoch 00180: early stopping
194/194 [==============================] - 0s 641us/step - loss: 0.5433 - mse: 0.5433
r2:  0.5957781636074844

<<standard scaling>>
Epoch 00106: early stopping
194/194 [==============================] - 0s 580us/step - loss: 0.5433 - mse: 0.5433
r2:  0.5957538298297659



1/25 << cnn >>

1.
Epoch 00177: early stopping
194/194 [==============================] - 0s 530us/step - loss: 0.3142 - mse: 0.3142
r2:  0.7662200945136256


1/26 << LSTM >>

Epoch 00210: early stopping
194/194 [==============================] - 0s 1ms/step - loss: 0.2948 - mse: 0.2948
r2:  0.780624729926207

<< conv 1d>>
Epoch 00049: early stopping
194/194 [==============================] - 1s 4ms/step - loss: 0.3600 - mse: 0.3600
r2:  0.732101630607719

'''