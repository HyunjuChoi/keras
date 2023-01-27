import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, Conv1D, Flatten
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

x = x.reshape(96, 4, 1)             #Rnn(LSTM)에 넣어줘야 하므로 3차원으로 변경
# y = y.reshape()

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.75, random_state=115)

x_train = x_train.reshape(72, 4, 1)
x_test = x_test.reshape(24, 4, 1)
x_predict = x_predict.reshape(7,4,1)


#2. modeling
model = Sequential()
# model.add(LSTM(128, input_shape=(4,1)))
# model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(4, 1)))              # Bidirectional => 방향 설정 (양방향)
# model.add(LSTM(64, activation='relu'))                                                      # return_sequences 추가하여 다음 레이어 LSTM 3차원 적용가능
model.add(Conv1D(100, 2, input_shape=(4,1)))                                                  # 레이어, 커널사이즈(1차원), 인풋 쉐잎
# model.add(Flatten())
model.add(Conv1D(64, 2, activation='relu'))
model.add(Conv1D(32, 2, activation='relu', padding='same'))
model.add(Conv1D(32, 2, activation='relu', padding='same'))
model.add(Conv1D(16, 2, activation='relu', padding='same'))
model.add(Conv1D(16, 2, activation='relu'))
# model.add(Dense(32, activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
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
model.fit(x, y, epochs=800, batch_size= 32 #, validation_split=0.2 
          #callbacks=[es]
          )


#4. evaluation, prediction
# loss = model.evaluate(x_test, y_test)
loss = model.evaluate(x, y)
print('loss: ', loss)

result = model.predict(x_predict)

print('예측값의 결과: ')

# print(result)

for i in range(0,7):
     print(i+100, ': ', result[i])


'''
<<Bidirectional>>
loss:  0.31530123949050903
예측값의 결과:  
[[101.05105 ]
 [101.95022 ]
 [102.82096 ]
 [103.66985 ]
 [104.49617 ]
 [105.299355]
 [106.07885 ]]
 
 
<< return sequences >>
loss:  0.005363041069358587
예측값의 결과: 
[[ 99.92091 ]
 [100.84354 ]
 [101.75373 ]
 [102.650696]
 [103.53357 ]
 [104.40152 ]
 [105.25377 ]]
 
 
<< Conv1D >>
1.
loss:  6.284637493081391e-06
예측값의 결과: 
[[100.001465]
 [101.00148 ]
 [102.00149 ]
 [103.0015  ]
 [104.00151 ]
 [105.00151 ]
 [106.00153 ]]
 
 2.
loss:  5.865533239557408e-06
예측값의 결과: 
100 :  [100.00117]
101 :  [101.00116]
102 :  [102.00115]
103 :  [103.00117]
104 :  [104.00114]
105 :  [105.00115]
106 :  [106.00112]
'''