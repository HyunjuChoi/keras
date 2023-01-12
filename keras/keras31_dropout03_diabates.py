
from sklearn.datasets import load_diabetes
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

path = 'C:/study/_data/ddarung/'
path2= 'C:/study/_save/'

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


#1. data
dataset = load_diabetes()
x = dataset.data                #(442, 10)
y = dataset.target              #(442, )

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, shuffle=True, random_state=115)

# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# #2. modeling
# model = Sequential()
# model.add(Dense(10, input_dim=10))
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
input1 = Input(shape=(10, ))
dense1 = Dense(10)(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(10)(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(20)(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(30)(drop3)
dense5 = Dense(40)(dense4)
dense6 = Dense(50)(dense5)
dense7 = Dense(40)(dense6)
dense8 = Dense(30)(dense7)
dense9 = Dense(20)(dense8)
dense10 = Dense(10)(dense9)
output1 = Dense(1)(dense10)

model = Model(inputs = input1, outputs = output1)

#3. compile training
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
es = EarlyStopping(
    monitor='loss',
    mode = 'min',
    patience = 20,
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


mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=1,
    save_best_only=True,                                              # save_best_only: 가중치 가장 좋은 지점 저장!
    # filepath= path + 'MCP/keras30_ModelCheckPoint3.hdf5' 
    filepath= filepath + 'k31_diabates_' + 'd_'+ date + '_'+ 'e_v_'+ filename                      #파일명 날짜, 시간 넣어서 저장하기            
)


model.fit(x_train, y_train, epochs=10000, batch_size=40, validation_split=0.2, verbose=1,
                 callbacks=[es, mcp])

model.save(path2 + 'keras31_dropout_save_model_diabates.h5')   

#4. evaluation prediction
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
#
print('r2: ', r2)


'''
#5. 시각화
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Diabates Loss')
plt.legend()
plt.show()
'''


'''결과치

1. patience = 10, monitor = 'loss'
Epoch 00051: early stopping
5/5 [==============================] - 0s 883us/step - loss: 3524.6282 - mse: 3524.6282
r2:  0.381446814979424


2. monitor = 'val_loss'

Epoch 00042: early stopping
5/5 [==============================] - 0s 749us/step - loss: 3409.4590 - mse: 3409.4590
r2:  0.4016584170307169


3. patience = 15
Epoch 00047: early stopping
5/5 [==============================] - 0s 1ms/step - loss: 3460.2988 - mse: 3460.2988
r2:  0.3927361977526439

4. monitor = 'loss'

Epoch 00072: early stopping
5/5 [==============================] - 0s 748us/step - loss: 3341.0562 - mse: 3341.0562
r2:  0.413662715249955

5.
Epoch 00071: early stopping
5/5 [==============================] - 0s 748us/step - loss: 3352.0669 - mse: 3352.0669
r2:  0.4117303329547157

6. patience = 20
Epoch 00119: early stopping
5/5 [==============================] - 0s 4ms/step - loss: 3440.9812 - mse: 3440.9812
r2:  0.39612635310893307



1/11일
<<min max>>
Epoch 00165: early stopping
5/5 [==============================] - 0s 756us/step - loss: 3490.1455 - mse: 3490.1455
r2:  0.38749835460257875

<<standard>>
Epoch 00104: early stopping
5/5 [==============================] - 0s 859us/step - loss: 3317.5781 - mse: 3317.5781
r2:  0.4177829363149138
'''