from tensorflow.keras.datasets import fashion_mnist

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()                  #텐서플로 mnist 데이터셋 불러와서 변수에 저장하기

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

# print(x_train, y_train)
print(x_train.shape, y_train.shape)                     # (60000, 28, 28) (60000,)=>(흑백데이터)
print(x_test.shape, y_test.shape)                       # (10000, 28, 28) (10000,)


print(x_train[0])
print(y_train[0])
'''
plt.imshow(x_train[115], 'Blues')
plt.show
'''

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Input, Flatten, MaxPooling2D, AveragePooling2D, Dropout

input1 = Input(shape=(28*28, ))
dense1 = Dense(128, activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(128, activation='relu')(drop1)
dense3 = Dense(64, activation='relu')(dense2)
dense4 = Dense(32, activation='relu')(dense3)
drop2 = Dropout(0.2)(dense4)
dense5 = Dense(32, activation='relu')(drop2)
dense6 = Dense(32, activation='relu')(dense5)
dense7 = Dense(16, activation='relu')(dense6)
drop3 = Dropout(0.2)(dense7)
dense8 = Dense(16, activation='relu')(drop3)
output1 = Dense(10, activation='softmax')(dense8)

model = Model(inputs = input1, outputs = output1)

 

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
    filepath= filepath + 'k38_dnn_fashion_' + 'd_'+ date + '_'+ 'e_v_'+ filename 
)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100, verbose=3, batch_size=128, validation_split=0.2, callbacks=[es, mcp])

#4. 평가 & 예측
results= model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])
print("Accuracy: %.2f%%" % (results[1]*100))


'''결과치
1/25

<<dnn 함수형 >>
Epoch 00040: early stopping
313/313 [==============================] - 0s 1ms/step - loss: 0.3789 - acc: 0.8739
loss:  0.3789461851119995
acc:  0.8738999962806702
Accuracy: 87.39%


2. <<drop out 추가>>
Epoch 00049: early stopping
313/313 [==============================] - 0s 1ms/step - loss: 0.4033 - acc: 0.8583
loss:  0.4033238887786865
acc:  0.858299970626831
Accuracy: 85.83%

3.
313/313 [==============================] - 0s 1ms/step - loss: 0.3889 - acc: 0.8687
loss:  0.38891953229904175
acc:  0.8687000274658203
Accuracy: 86.87%

'''