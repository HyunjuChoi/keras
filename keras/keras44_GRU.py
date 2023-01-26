import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping

#1. data
dataset = np.array([1,2,3,4,5,6,7,8,9,10])          #(10, )

# y 데이터가 따로 없기 때문에 dataset에서 직접 x data와 y data로 나눈다
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9]])
y = np.array([4,5,6,7,8,9,10])

# print(x)

print(x.shape, y.shape)             # (7, 3) (7,)

#rnn의 특징인 "연결 순환 구조" 모델링을 위해 데이터 reshape (dnn과의 차이점!)
x = x.reshape(7,3,1)                # 3개씩 묶은 전체 7개의 데이터 => 묶음 데이터 1개 => 1개씩 순차적 계산 
                                    # [[[1],[2],[3]], [[2],[3],[4]], ... ]           
                                    
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.7, random_state=115)

#2. modeling
model = Sequential()
# model.add(SimpleRNN(64, input_shape=(3,1)))                     # (N, 3, 1)  => ([batch, timesteps, feature])   
#                                                                 : timesteps(input lenght) 만큼 자른 후 feature크기 만큼 수행
# model.add(LSTM(units=64, input_length=3, input_dim=1))
# model.add(LSTM(64, input_shape=(3,1)))
model.add(GRU(64, input_shape=(3,1)))
model.add(Dense(32, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


model.summary()

# model.summary()
# simpleRNN: 4224
# LSTM: 16896               (simpleRNN * 4)
# GRU: 12684 => 3 * units * (feature + bias + units + ?) = parameters   (gru 최근 업데이트 돼서 파라미터 계산할 때 하나 더 추가됨)
