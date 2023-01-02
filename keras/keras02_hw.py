import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#[13] 예측값 구하기

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))

model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=2000)

result = model.predict([13])
print('result: ', result)



