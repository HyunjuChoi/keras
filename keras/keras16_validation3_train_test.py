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

#[실습] train_test_split으로 자르기!
#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.87, test_size=0.13, random_state=115)
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=115)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.67, random_state=115)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=115)


print(x_train, x_test, y_train, y_test, x_val, y_val)

'''
#2. modeling
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. compile and training
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_data=(x_val, y_val))         #검증 데이터 추가!!!

#4. evaluation, prediction
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)

result = model.predict([17])
print('17 예측값: ', result)


loss:  2.1110454326844774e-06
17 예측값:  [[16.996725]]

loss:  0.00022585219994653016
17 예측값:  [[16.898645]]

loss:  87.93248748779297
17 예측값:  [[2.29396]]

loss:  1.368147786706686e-05
17 예측값:  [[16.991247]]

'''