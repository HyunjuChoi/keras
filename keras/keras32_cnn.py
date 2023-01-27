from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten          # Conv2D: 2차원 이미지 데이터 다루기 위한 import
                                                                    # flatten: 차원 펼치기

model = Sequential()
                                                        # input_shape = (5, 5, 1)인 각기 다른 이미지 데이터 if, 60,000장 = > (60000,    5,    5,    1) 
                                                                                                                    # = (batch_shape, row, col, channels )
model.add(Conv2D(filters=10, kernel_size=(2,2),         # kernel_size: 조각낼 이미지 사이즈, 
                                                        # filters = 10: 조각낸 이미지 수치(장 수) 10장으로 늘린다.
                 input_shape=(5, 5, 1)                  # 원본 이미지 사이즈: 5 X 5의 1장(흑백 이미지)         
))                                                      #  => (60000, 4, 4, 10)으로 바뀜. 맨 앞의 장 수(데이터 개수)는 바뀌지 않기 때문에
                                                        # (N, 4, 4, 10)으로 표현 가능

'''
Input_shape: data_format = 'channel_first'면 batch_shape + (channels, row, columns), batch_shape: 훈련 단위
             data_format = 'channel_last'면 batch_shape + (row, column, channels): default
'''
model.add(Conv2D(5, (2,2)))
# model.add(Conv2D(filters=5, kernel_size=(2,2)))       # filter 수는 직접 수행해보며 적절한 값 찾는다 (다음 레이어의 input? 노드 같은 개념)
                                                        # => (N, 3, 3, 5)
model.add(Conv2D(7, (2, 2)))
model.add(Conv2D(6, 2))                                 #Conv2D는 2차원 다루는 거라 (2,2)나 그냥 2나 똑같이 인식함
model.add(Flatten())                                    # Dense 적용해 주기 위해 차원 펼치기 수행 => 3*3*5 = (N, 45)
model.add(Dense(units=10))                              # (N, 10),  input = (batch_size, input_dim), input_dim = column의 개수
model.add(Dense(1, activation='relu'))                                     # (N, 1)

model.summary()