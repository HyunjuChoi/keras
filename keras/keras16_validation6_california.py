'''
[실습]
r2 0.55~0.6 이상
'''


from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

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

#2. modeling
model = Sequential()
model.add(Dense(1, input_dim=8))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. compile and training
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=100,
          validation_split=0.3)
end = time.time()


#4. evaluation prediction
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
print('r2: ',r2)

print('time: ', end-start)



'''
1.
model = Sequential()
model.add(Dense(1, input_dim=8))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. compile and training
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=10000, batch_size=500)

=>r2:  0.5882458818645178


2. relu 적용, validation_set 설정, 
r2:  -0.0016761227565078585
time:  44.03359031677246


3. batch_size= 100
r2:  0.6466477577634997
time:  121.37677597999573

'''


#tf274gpu:  148.33980655670166
#tf27:  60.83969163894653