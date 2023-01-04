#실습
#1. R2를 음수가 아닌 0.5 이하로 줄이기
#2. 데이터는 건들지 말 것
#3. 레이어는 인풋 아웃풋 포함 7개 이상
#4. batch_size = 1
#5. 히든레이어외 노드는 각각 10개 이상 100개 이하
#6. train 70%
#7. epochs = 100번 이상
#8. loss 지표는 mae 또는 mse


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))          #변수 두개 받아서 mse 처리=> 제곱근 값으로 리턴


#1. data
x = np.array(range(1,21))
y = np.array(range(1,21))

print(x.shape)
print(y.shape)
# print(z.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=824)

#2. modeling
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(1))

#3. compile training
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1)


#4. eva pre
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)           #x_test와 y_test 비교해서 정확도 확인

print('x: ',x_test)
    
print("y:",y_predict)

# y_predict = y_predict/2

#RMSE(y_test, y_predict)

r2 = r2_score(x_test,y_predict)


'''
print('====================')
print(y_test)
print(y_predict)
print('RSME: ', RMSE(y_test, y_predict))
'''
print('====================')
print('R2: ', r2)
print('====================')

#R2:  0.6498179084012543        => 대충 64%정도 정확하다는 의미


