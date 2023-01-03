import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x = np.array([1,2,3,4,5,6,7,8,9,10])        #(10, )
y = np.array(range(10))                     #(10, )       => weight = 1, bias = -1

#numpy 리스트 슬라이싱
x_train = x[:7]             #index랑 data 값 헷갈리지 않기 / index= 0 ~ n-1
x_test = x[7:]

y_train = y[:7]             #y_train = y[:-2]
y_test = y[7:]              #y_test = y[-3:]


'''
print(x_train)
print(x_test)
print(y_train)
print(y_test)
'''

#2. modeling
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(1))


#3. compile training
model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train, batch_size=1, epochs=100)         #훈련용 데이터로 fit


#4. evalu predict
loss = model.evaluate(x_test,y_test)            #평가용 데이터로  evaluate
print('loss: ', loss)
result = model.predict([11])
print('result: ', result)


