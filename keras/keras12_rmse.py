import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))          #변수 두개 받아서 mse 처리=> 제곱근 값으로 리턴

#1. data
x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])
#z = np.array([[[1],[2],[3]]] )

print(x.shape)
print(y.shape)
# print(z.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123
)

#2. modeling
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. compile training
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1)


#4. eva pre
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)           #x_test와 y_test 비교해서 정확도 확인

# RMSE(y_test, y_predict)

print('====================')
print(y_test)
print(y_predict)
print('RSME: ', RMSE(y_test, y_predict))
print('====================')


# loss:  [14.75754165649414, 2.949686050415039] => 두가지인 이유! loss값과 metrics값 같이 나오는 것~

#RSME:  3.8568779526248305
#       3.8596254016045326
#       3.845519261883102           =>결과치 값 중 가장 좋은 것 저장해서 쓰도록 하자