
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
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

# #2. modeling
# model = Sequential()
# model.add(Dense(1, input_dim=8))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(20))
# model.add(Dense(30))
# model.add(Dense(40))
# model.add(Dense(50))
# model.add(Dense(40))
# model.add(Dense(30))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))

#2. modeling
input1 = Input(shape=(8, ))
dense1 = Dense(10)(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(20)(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(30)(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(40)(drop3)
dense5 = Dense(50)(dense4)
dense6 = Dense(40)(dense5)
dense7 = Dense(30)(dense6)
dense8 = Dense(20)(dense7)
dense9 = Dense(10)(dense8)
output1 = Dense(1)(dense9)

model = Model(inputs=input1, outputs=output1)

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
mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=1,
    save_best_only=True,                                              # save_best_only: 가중치 가장 좋은 지점 저장!
    # filepath= path + 'MCP/keras30_ModelCheckPoint3.hdf5' 
    filepath= filepath + 'k31_cali_' + 'd_'+ date + '_'+ 'e_v_'+ filename                      #파일명 날짜, 시간 넣어서 저장하기            
)

model.fit(x_train, y_train, epochs=1000, batch_size=500, validation_split=0.2, verbose=1,
                callbacks=[es, mcp])

model.save(path + 'keras31_dropout_save_model_california.h5')   

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

'''