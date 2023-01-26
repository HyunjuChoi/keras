import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
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
model.add(SimpleRNN(64, input_shape=(3,1)))                     # (N, 3, 1)  => ([batch, timesteps, feature])   : timesteps 만큼 자른 후 feature크기 만큼 수행
model.add(Dense(32, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


model.summary()


'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 64)                4224

 dense (Dense)               (None, 32)                2080

 dense_1 (Dense)             (None, 32)                1056

 dense_2 (Dense)             (None, 16)                528

 dense_3 (Dense)             (None, 8)                 136

 dense_4 (Dense)             (None, 1)                 9

=================================================================
Total params: 8,033
Trainable params: 8,033
Non-trainable params: 0

'''


# simple_rnn param# 4224인 이유!   => recurrent_weigths:(64 * 64) + input_weights:(1 * 64) + biases:(64) = 4224
#                                 =>  64 * (64 + 1 + 1) = 4224
#                                 => units(=batch) * (feature + bias + units) = param 


"""
[참고링크] https://stackoverflow.com/questions/50134334/number-of-parameters-for-keras-simplernn

This number represents the number of trainable parameters (weights and biases) in the respective layer, in this case your SimpleRNN.

>>> recurrent_weights = num_units*num_units <<<

>>> input_weights = num_features*num_units <<<

>>> biases = num_units*1 <<<

So finally we have the formula:

>>>> recurrent_weights + input_weights + biases <<<

or

num_units* num_units + num_features* num_units + biases

=(num_features + num_units)* num_units + biases

"""