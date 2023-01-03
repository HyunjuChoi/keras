import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. modeling
model = Sequential()                        #sequential model 정의
#deep learning 구현
model.add(Dense(3, input_dim=1))            #input layer
model.add(Dense(50))                         #hidden layer    
model.add(Dense(40))                         #(input layer 제외하고 INPUT data 명시 안해도 됨)
model.add(Dense(20))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
model.add(Dense(20))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
model.add(Dense(1))                         #output layer

#3. compile, Training
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=10, batch_size=1)    #adjust batch size                   

#4. prediction
result = model.predict([6])
print("6 result: ", result)
