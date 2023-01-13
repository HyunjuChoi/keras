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

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28 ,1)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print(np.unique(y_train, return_counts=True))               #output class 개수 확인!!! (마지막 layer에 설정해줘야 하니까)
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64)) 
# => class개수 [0~9]까지 
# 10개, 0~9까지 일정하게 되어있으니까 ohe 안해도 된다... 뭔솔???? 


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(filters=128, kernel_size = (2,2), input_shape=(28, 28, 1), 
                 activation='relu'))                            # (27, 27, 128)

model.add(Conv2D(filters=64, kernel_size = (2,2)))              # (26, 26, 64)
model.add(Conv2D(filters=64, kernel_size = (2,2)))              # (25, 25, 32)
model.add(Flatten())                                            # 25*25*32 = 40,000
model.add(Dense(32, activation='relu'))                         # input_shape = (60000, 40000)에서 '행 무시'이므로 (40000, )
                                                                # 60000 = batch_size, 40000 = input_dim
                                                                
model.add(Dense(10, activation='softmax'))                      #output 노드 10개이므로 다중분류! 

#3. 컴파일 & 훈련

es = EarlyStopping(
     monitor='val_loss',
     mode='min',
     patience=50,
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
    filepath= filepath + 'k34_mist_' + 'd_'+ date + '_'+ 'e_v_'+ filename 
)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100, verbose=3, batch_size=128, validation_split=0.3, callbacks=[es, mcp])

#4. 평가 & 예측
results= model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])



'''
1/13 결과치
<<es, mcp 적용>>

Epoch 00054: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.1078 - acc: 0.9705
loss:  0.10780102759599686
acc:  0.9704999923706055


2. <<kernel_size= (2,2) -> (3,3), filters = 64->32>>
loss:  2.3010241985321045
acc:  0.11349999904632568 ....

3. <<batch_size = 128, kernel, filters 1번으로 원상복귀;;>>
Epoch 00066: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.1313 - acc: 0.9714
loss:  0.13126905262470245
acc:  0.9714000225067139


4. 
Epoch 00071: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.1502 - acc: 0.9727
loss:  0.15016406774520874
acc:  0.9726999998092651

5.Epoch 00072: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.1457 - acc: 0.9745
loss:  0.1457120180130005
acc:  0.9745000004768372

6.Epoch 00061: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.1454 - acc: 0.9713 
loss:  0.1453816294670105
acc:  0.9713000059127808

'''