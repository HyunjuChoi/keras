import numpy as np
#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
'''input dimension = 2, output dimension = 1'''

x = np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])           #데이터셋 두개
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

print(x.shape)      #행렬(데이터)의 구조: (2,10) = > 2X10 행렬 상태, 10개의 스칼라를 가진 벡터 2개
                    # => 10X2로 만들려면 원래는 [[1,1], [2,1] ... ]의 형태로 만들어야 됨
print(y.shape)      #(10,)          => 10개의 스칼라를 가진 벡터 1개

x = x.T             #행,열 바꿈
print(x.shape)      # (10,2)
''''
#2. modeling
model = Sequential()
model.add(Dense(5, input_dim=2))      #인풋데이터 = 2개 (데이터셋 개수), 5개 아웃풋, 열의 개수=input_dim
                                      #행 무시, 열 우선
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))


#3. compile, training
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=100, batch_size=1)

#4. evaluation, Prediction
loss = model.evaluate(x,y)                  #배치 최대 사이즈 = 32, 통상 배치 명시 안해도 됨
print('loss: ', loss)
result = model.predict([[10, 1.4]])
print('[10, 1.4]의 result: ',result)
'''
'''
결과: 여러번 돌렸을 때 loss와 result 제일 좋았던 값 기록하기


'''