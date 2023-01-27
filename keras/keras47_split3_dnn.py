
##### dnn #####


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler, MinMaxScaler

a = np.array(range(1,101))
x_predict = np.array(range(96, 106))
#예상 y = 100 ~ 106

# print(x_predict.shape)              # (10,)

size = 5        # x는 4개 y는 1개

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size +1):
        subset= dataset[i : (i + size)]
        aaa.append(subset)
    #print(type(subset))    
    return np.array(aaa)                    # 모델링 할 때 주로 이용하는 데이터 형태로 바꿔주기 위함

bbb = split_x(a, size)          # 데이터 5개씩 자름

x = bbb[:,:-1]                  # x는 앞 4개
y = bbb[:,-1]                   # y는 뒤 1개

# x_predict = split_x(x_predict, size)          # (6, 5)          => (7,4)로 해야 됨

size2 = 4

x_predict = split_x(x_predict, size2)

# ccc= x_predict[:, :-1]
# ddd = x_predict[:, -1]

print(x_predict)
# print(ccc)
#ccc = ccc.reshape(6, 4, 1)

# print(x_predict, x_predict.shape)

# print(x.shape, y.shape)                     # (96, 4) (96,)

# x_predict = np.array([97,98,99,100])

#### 실습: LSTM 모델 구성 ####

# x = x.reshape(96, 4, 1)             #Rnn(LSTM)에 넣어줘야 하므로 3차원으로 변경
# y = y.reshape()

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.75, random_state=115)

# x_train = x_train.reshape(72, 4, 1)
# x_test = x_test.reshape(24, 4, 1)
# x_predict = x_predict.reshape(7,4,1)


#2. modeling
model = Sequential()
model.add(Dense(128, input_shape=(4, )))
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
model.fit(x, y, epochs=500, batch_size= 32 #, validation_split=0.2 
          #callbacks=[es]
          )


#4. evaluation, prediction
loss = model.evaluate(x, y)
# loss = model.evaluate(x, y)
print('loss: ', loss)

result = model.predict(x_predict)

print('예측값의 결과: ', result)