import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x = np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
            [9,8,7,6,5,4,3,2,1,0]])           #데이터셋 3개

y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

print(x.shape)
print(y.shape)

x = x.T
print(x.shape)

#2. modeling
model = Sequential()
model.add(Dense(5, input_dim=3))                #input data set이 3차행렬이므로 3으로 해줘야 오류 안남
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))                 # Q. 아웃풋 레이어는 항상 1이어야 할까? => 아닌듯?!

#3. compile & training
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=100,batch_size=1)

#4. evaluation, Prediction
loss = model.evaluate(x,y)
print('loss: ', loss)
result= model.predict([[10, 1.4, 0]])
print('[10, 1.4, 0]의 result: ', result)


'''
최적결과값
loss:  0.06535043567419052
[10, 1.4, 0]의 result:  [[20.129112]] 

loss:  0.15761856734752655
[10, 1.4, 0]의 result:  [[20.091978]]

loss:  0.141574427485466
[10, 1.4, 0]의 result:  [[19.916939]]

==> 예측값과 loss의 최적값이 서로 다를 때 
    => loss값이 우선이기 때문에 loss 값 더 좋은 걸로 선택!

'''