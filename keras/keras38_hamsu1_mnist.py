import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()                  #텐서플로 mnist 데이터셋 불러와서 변수에 저장하기

# print(x_train, y_train)
print(x_train.shape, y_train.shape)                     # (60000, 28, 28) (60000,)=>(흑백데이터)
print(x_test.shape, y_test.shape)                       # (10000, 28, 28) (10000,)

# 이미지 데이터는 (데이터, 행, 열, 컬러)로 4차원로 구성, input_shape = 데이터 빼고 3차원.
# 현재 x_train, x_test 데이터 3차원이므로 4차원으로 늘려줘야함

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28 ,1)

#reshape
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print(np.unique(y_train, return_counts=True))               #output class 개수 확인!!! (마지막 layer에 설정해줘야 하니까)
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64)) 
# => class개수 [0~9]까지 
# 10개, 0~9까지 일정하게 되어있으니까 ohe 안해도 된다... 뭔솔???? 



#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Input, Flatten, MaxPooling2D, AveragePooling2D, Dropout

input1 = Input(shape=(784, ))
dense1 = Dense(120, activation='relu')(input1)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(32, activation='relu')(dense2)
dense4 = Dense(32, activation='relu')(dense3)
dense5 = Dense(32, activation='relu')(dense4)
dense6 = Dense(16, activation='relu')(dense5)
output1 = Dense(10, activation='softmax')(dense6)

model = Model(inputs=input1, outputs=output1)
#model.summary()


#3. 컴파일 & 훈련

es = EarlyStopping(
     monitor='val_loss',
     mode='min',
     patience=20,
     restore_best_weights=True,           
     verbose=3
)

import datetime
date = datetime.datetime.now()
print(date)
print(type(date))                           # <class 'datetime.datetime'>
date= date.strftime("%m%d_%H%M")

print(date)                                 

filepath = 'C:/study/_save/MCP/'
filename = '{epoch:04d}-{val_loss: .4f}.hdf5'                       # d: digit, f: float 


mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=3,
    save_best_only=True,
    filepath= filepath + 'k38_dnn1_' + 'd_'+ date + '_'+ 'e_v_'+ filename 
)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
model.fit(x_train, y_train, epochs=200, verbose=3, batch_size=100, validation_split=0.2, callbacks=[es, mcp])

#4. 평가 & 예측
results= model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])
print("Accuracy: %.2f%%" % (results[1]*100))




'''
1/25

<dnn 함수형 모델>

1.Epoch 00039: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.1156 - acc: 0.9737
loss:  0.11559653282165527
acc:  0.9736999869346619
Accuracy: 97.37%



'''