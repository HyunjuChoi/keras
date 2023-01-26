import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler, MinMaxScaler

a = np.array(range(1,11))
timesteps = 5

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps +1):
        subset= dataset[i : (i + timesteps)]
        aaa.append(subset)
    #print(type(subset))    
    return np.array(aaa)                    #모델링 할 때 주로 이용하는 데이터 형태로 바꿔주기 위함

bbb = split_x(a, timesteps)

print(bbb)

x = bbb[:,:-1]
y = bbb[:,-1]

# print(x.shape, y.shape)             #(6, 4) (6,)

x_predict = np.array([7,8,9,10])

#### 실습: LSTM 모델 구성 ####

x = x.reshape(6, 4, 1)
# y = y.reshape()

# x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.7, random_state=115)

#2. modeling
model = Sequential()
model.add(LSTM(128, input_shape=(4, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))


#3. compile and training
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=1000, batch_size= 1, validation_split=0.2, callbacks=[es])
model.fit(x, y, epochs=1000, batch_size= 1, validation_split=0.2, callbacks=[es])


#4. evaluation, prediction
# loss = model.evaluate(x_test, y_test)
loss = model.evaluate(x, y)
print('loss: ', loss)

y_predict = x_predict.reshape(1, 4, 1)

result = model.predict(y_predict)

print('[7,8,9,10]의 결과: ', result)