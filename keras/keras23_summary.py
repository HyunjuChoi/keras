from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. data
x = np.array([1,2,3])
y = np.array([1,2,3])


#2. modeling
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary()                         #model architecture의 structure와 연산량 출력
#=> bias가 추가되어 계산된 연산량(parameter)이 나옴
