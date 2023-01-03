import tensorflow as tf
import numpy as np

print (tf.__version__)



#1. Data 준비
x = np.array([1,2,3])               # numpy 형식의 행렬 
y = np.array([1,2,3])              

#2. model 구성(케라스 문법을 통한 텐서플로)
from tensorflow.keras.models import Sequential      # 딥러닝 순차모델         
from tensorflow.keras.layers import Dense           # y = wx+b 구성 위한 기초

model = Sequential()                       # Sequential model 제작, layer에 순차적으로 연산한다
                                           # Dense 레이어 넣음 
model.add(Dense(1, input_dim=1))           # 1= output_dim(y, 출력), input_dim(x 행렬) 한덩어리 1로 추가

#최적의 weight와 bias 찾기 위한 "정제된" 데이터가 중요!!!


#3. compile과 훈련
model.compile(loss='mae', optimizer='adam')           # mean absolute error => loss  값을 낮추기 위한 기준
                                                      # adam: loss 최적화 하기 위한 넘
model.fit(x,y, epochs=300)                             # fit: 모델 훈련시키기, 0에 수렴할 수 있도록 훈련 많이 시켜야 함
                                                      # epochs: 훈련횟수
                                                      
#4. 평가, 예측
result = model.predict([4])
print('result: ', result)
