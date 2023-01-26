import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler, MinMaxScaler

#1.data

x = np.array([[1,2,3], [2,3,4],[3,4,5],
              [4,5,6],[5,6,7],[6,7,8],
              [7,8,9],[8,9,10],[9,10,11],
              [10,11,12],[20,30,40],
              [30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

y_predict = np.array([50,60,70])

# print(x.shape, y.shape)                 # (13, 3) (13,)   

x = x.reshape(13,3,1)


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size = 0.7, random_state=115)

'''
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x_train = scaler.transform(x_train)             # x의 값 변환하여 x에 저장
x_test = scaler.transform(x_test)  

x = x.reshape(13,3,1)

# print(x_train.shape, x_test.shape)              # (9, 3) (4, 3)

x_train.reshape(9,3,1)
x_test.reshape(4,3,1)

# x_predict = x_predict.reshape()
'''

#2. modeling
model = Sequential()
model.add(LSTM(128, input_shape=(3,1),                                  # (N, 3, 1) => (N, 128)   : LSTM은 3차원에서 2차원으로 던져줌 
               return_sequences=False))                                 # (그래서 그냥 던지면 다음 레이어가 LSTM일 때 3차원이 아닌 2차원으로 받아서 에러남)
                                                                        # => 'return_sequences = True' 추가 해줘야 함~~!
                                                                        
# model.add(LSTM(64, activation='relu'))             
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


#3. compile, training

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience = 50,
    restore_best_weights=True,
    verbose=1
)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es])

#4. evaluation, prediction
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_pred = y_predict.reshape(1,3,1)

# print(y_pred)

result = model.predict(y_pred)
print('[50,60,70]의 예측값: ', result)