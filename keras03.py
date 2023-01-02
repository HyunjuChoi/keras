import numpy as np
import tensorflow as tf

print(tf.__version__)               #2.7.4


#1. (정제된) data
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

#2. modeling     y=wx+b
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))

#3. compile, training
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=200)

#4. 평가, 예측
result=model.predict([6])
print('6 예측: ',result)
