
from sklearn.datasets import load_diabetes
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping


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
earlyStopping = EarlyStopping(
    monitor='loss',
    mode = 'min',
    patience = 20,
    restore_best_weights=True,
    verbose=1
)
hist = model.fit(x_train, y_train, epochs=10000, batch_size=40, validation_split=0.2, verbose=1,
                 callbacks=[earlyStopping])

#4. evaluation prediction
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

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

'''