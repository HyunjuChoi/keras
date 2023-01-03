import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x = np.array([range(10), range(21, 31), range(201, 211)]) 
y = np.array([[1,2,3,4,5,6,7,8,9,10],
                [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4] ])               # => x 특성 개수와  y 특성 개수 달라도 가능

# print(x.shape)          #(3, 10)

x = x.T
#print(x.shape)
y = y.T

#2. modeling
model = Sequential()
model.add(Dense(3, input_dim=3))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))                 #아웃풋 개수 설정!!! (ex. 2개 원하면 2, 1개 원하면 1)

#3. compile , Training
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)

#4. evaluation, prediction
loss = model.evaluate(x,y)
print('loss: ', loss)
result = model.predict([[9,30,210]])
print('result: ', result)


'''
결과치

loss:  0.13469812273979187
result:  [[10.012307   1.7830862]]

loss:  0.5101258158683777
result:  [[9.630806 2.148257]]

loss:  0.20242038369178772
result:  [[10.220987   1.7377214]]

loss:  0.29532068967819214
result:  [[9.258127  1.4169551]]

<히든레이어 더 추가>

loss:  0.2715838849544525
result:  [[9.977246 1.849498]]

loss:  0.4599735140800476
result:  [[9.539481  1.9386597]]

loss:  0.1777869313955307
result:  [[10.221766   1.6264371]]

loss:  0.1191834956407547
result:  [[10.071613   1.6455628]]

'''
