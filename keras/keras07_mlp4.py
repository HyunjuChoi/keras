import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x = np.array([range(10)]) 
y = np.array([[1,2,3,4,5,6,7,8,9,10],
                [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
                [9,8,7,6,5,4,3,2,1,0]])               # => x 특성 개수와  y 특성 개수 달라도 가능 but 인풋 대비 아웃풋 개수가 더 많으면 성능 구려짐

# print(x.shape)          #(1, 10)

x = x.T
#print(x.shape)
y = y.T

#2. modeling
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(50))
model.add(Dense(3))                 #아웃풋 개수 설정!!! (ex. 2개 원하면 2, 1개 원하면 1)

#3. compile , Training
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)               #훈련 x,y

#4. evaluation, prediction
loss = model.evaluate(x,y)                              #평가 x,y => 같은 걸로 훈련하고 평가하면 안돼~!
                                                        #훈련 시키지 않은 값으로 평가해야함. 그것이 예측!
print('loss: ', loss)
result = model.predict([9])             #행 무시, 열 우선! 때문에 예측 시 열의 개수 동일하게 맞춰주기~~!
print('result: ', result)


'''
결과치   [10, 1.4, 0]

loss:  1.2202162742614746
result:  [[10.254808   1.5246379  3.7304811]]

loss:  0.1016954630613327
result:  [[ 9.825953    1.5559833  -0.02050781]]

loss:  0.1584232747554779
result:  [[9.59699    1.5873622  0.24809659]]

loss:  0.11686916649341583
result:  [[9.835254   1.517583   0.23153469]]

loss:  0.07201896607875824
result:  [[9.963491   1.6203778  0.10701638]]

loss:  0.09272646903991699
result:  [[9.947482   1.5573055  0.21219406]


히든레이어 추가

loss:  0.1333228051662445
result:  [[9.882494  1.6048012 0.290726 ]]

loss:  0.1637945920228958
result:  [[ 9.891642    1.5238612  -0.40118307]]

loss:  0.05402180552482605
result:  [[ 9.9753742e+00  1.6791238e+00 -2.4198592e-03]]


레이어 Dense 변경

loss:  0.07533735036849976
result:  [[9.98368    1.5553927  0.12370464]]


배치크기 변경

loss:  0.09555286914110184
result:  [[10.0333605   1.5483954   0.24111612]]

'''
