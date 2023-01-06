'''
[과제, 실습]
r2 0.62 이상
'''

from sklearn.datasets import load_diabetes
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


#1. data
dataset = load_diabetes()
x = dataset.data                #(442, 10)
y = dataset.target              #(442, )

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, shuffle=True, random_state=115)


#2. modeling
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. compile training
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=5000, batch_size=5, validation_split=0.3)

#4. evaluation prediction
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('r2: ', r2)


'''
결과

1.
model = Sequential()
model.add(Dense(1, input_dim=10))
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

#3. compile training
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=1000, batch_size=10)

r2:  0.41391420912937316

2. batch_size = 5
r2:  0.41539187338101724

3. epochs=100, batch_size=1
r2:  0.41238776521742326

4.
epochs=1000, batch_size=1
r2:  0.4090679266539592


5. epochs=5000, batch =1 
r2:  0.4131086087913519

6. epochs = 10000, batch = 40
r2:  0.41288371438211735



7. relu, validation_set 설정, epochs= 1000
r2:  0.39170024241738977


8. epochs = 5000
r2:  0.05833267850797785

9.
r2:  0.3004811908780033

10. batch_size = 20으로 바꿈
r2: -어쩌구저쩌구 나옴

11. epochs = 5000, batch_size=5
r2:  -0.02224648549788988

''' 