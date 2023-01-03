from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

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
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1)


#4. eva pre
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x)


#시각화
import matplotlib.pyplot as plt                 #그래프 그려주는 라이브러리

plt.scatter(x,y)                                #실제 데이터 점 찍기
plt.plot(x, y_predict, color='red')             #가중치 구해서 나온 예측값 그래프 그리기
plt.show()


'''
'
'
'
14/14 [==] - 0s 1ms/step - loss: 2.0595
Epoch 199/200
14/14 [==] - 0s 1ms/step - loss: 1.8548
Epoch 200/200
14/14 [==] - 0s 1ms/step - loss: 1.9012       =>훈련 데이터는 loss가 좋지만
1/1 [==] - 0s 170ms/step - loss: 3.0800       =>예측 데이터의 loss는 훈련 데이터에 비해 loss 값 나쁨

'''