import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


#1. data
'''
x_train = np.array((range(1,11)))
y_train = np.array((range(1,11)))
x_test = np.array([11, 12, 13])
y_test = np.array([11, 12, 13])

#x_train = x_train.T
#y_train = y_train.T

#검증데이터 추가~! (machine이 훈련한 것을 검증한다)
x_validation = np.array([14, 15, 16])
y_validation = np.array([14, 15, 16])
'''

x = np.array(range(1,17))
y = np.array(range(1,17))

#print(x, y)

#[실습] slicing으로 자르기!
x_train = x[:10]
x_test = x[10:13]
y_train = y[:10]
y_test = y[10:13]

x_validation = x[13:]
y_validation = y[13:]

#print(x_train, x_test, y_train, y_test, x_validation, y_validation)


#2. modeling
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. compile and training
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_data=(x_validation, y_validation))         #검증 데이터 추가!!!

#4. evaluation, prediction
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

result = model.predict([17])
print('17 예측값: ', result)

'''
loss:  8.259498827101197e-06
17 예측값:  [[16.994423]]

loss:  2.382876118645072e-10
17 예측값:  [[16.999971]]

loss:  0.0007036701426841319
17 예측값:  [[16.913937]]

'''