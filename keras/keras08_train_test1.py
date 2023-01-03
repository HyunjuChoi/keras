import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
# x = np.array([1,2,3,4,5,6,7,8,9,10])        #(10, )
# y = np.array(range(10))                     #(10, )       => weight = 1, bias = -1

x_train = np.array([1,2,3,4,5,6,7])           #(7, )
x_test = np.array([8,9,10])                   #(3, )   => 한개의 특성 가지기 때문에 데이터 셋 바뀌어도 괜찮(행은 무시)

y_train = np.array(range(7))
y_test = np.array(range(7,10))



#2. modeling
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(1))


#3. compile training
model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train, batch_size=1, epochs=100)         #훈련용 데이터로 fit


#4. evalu predict
loss = model.evaluate(x_test,y_test)            #평가용 데이터로  evaluate
print('loss: ', loss)
result = model.predict([11])
print('result: ', result)


'''
결과

1.히든레이어 추가 + epochs=100

loss:  0.18031089007854462
result:  [[10.219087]]

loss:  0.18433792889118195
result:  [[10.217776]]

loss:  0.03518867492675781
result:  [[10.039094]]


>>dense 수치 앞뒤로 크게 차이나면 성능 저하됨

loss:  0.07367102056741714
result:  [[10.086369]]


loss:  0.011727173812687397
result:  [[10.009757]]





'''
