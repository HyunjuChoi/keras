#<넘파이와 텐서플로우 가져오기>
import numpy as np #numpy를 가져오고 줄여서 np라 한다.
import tensorflow as tf #tensorflow를 가져오고 줄여서 tf라 한다.
print(tf.__version__)   # 2.7.4 #tf의 version을 출력한다.

#1. (정제된) 데이터
x = np.array([1,2,3,4,5]) 
y = np.array([1,2,3,5,4])

#2. 모델구성 y-wx+b
from tensorflow.keras.models import Sequential
#tensorflow에 있는 keras 문법을 사용해 sequential 순차모델을 가져온다.
#keras 모델에는 Sequential 모델과 함수 API에 사용되는 Model이 있다.
#Sequential은 레이어를 선형으로 연결하여 구성하는, 케라스에서 가장 단순한 신경망 모델이다.
#Model은 케라스 함수 API로 다중 입력/다중 출력 모델, 방향성 비순환 그래프(DAG), 공유된 계층이 있는 모델과 같은 복잡한 모델을 만드는데 유용하다.

from tensorflow.keras.layers import Dense
#tensorflow에 있는 keras 문법을 사용해 Dense 계층(레이어)를 가져온다.
#keras 계층에는 Dense, Activation, Dropout, Lambda 등이 있다.

model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
#compile이란 사람의 언어를 컴퓨터가 이해할 수 있는 언어로 바꿔주는 과정을 말한다.
#loss 오차를 구하는 방법으로는 mse와 mae가 있다.
#mse=mean squared error = 평균 제곱 오차 = 오차의 제곱에 대한 평균 : 손실함수
#mae=mean absolute error = 평균 절대 오차 = 모든 절대 오차의 평균 : 회귀지표
#https://dbstndi6316.tistory.com/297

model.fit(x, y, epochs=200)
#모델을 학습시키는 걸 fit이라 표현한다.

#4. 평가, 예측
result = model.predict([6])
print('6의 예측값 : ', result)

"""
결과
Epoch 200/200
1/1 [==============================] - 0s 0s/step - loss: 0.4006
6의 예측값 :  [[5.9978437]]
"""