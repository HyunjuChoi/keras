from tensorflow.keras.datasets import cifar10, cifar100                 #컬러 이미지 데이터
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()                  #텐서플로 mnist 데이터셋 불러와서 변수에 저장하기

# print(x_train, y_train)
print(x_train.shape, y_train.shape)                     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)                       # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))           # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
                                                        # array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))

x_train = x_train/255.                                  #이거 뭐야?
x_test = x_test/255.


                                                        
#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, AveragePooling2D

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(32, 32, 3),
                 activation='relu', padding='same'))                    # (31, 31, 128)
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same'))        # (30, 30, 64)
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(2,2), padding='same'))        # (29, 29, 64)
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Conv2D(filters=16, kernel_size=(2,2), padding='same'))        # (28, 28, 32)
model.add(AveragePooling2D())
model.add(Flatten())                                    # 25,088
model.add(Dense(10, activation='softmax'))

#3. 컴파일 & 훈련

es = EarlyStopping(
     monitor='val_loss',
     mode='min',
     patience=20,
     restore_best_weights=False,           
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
    filepath= filepath + 'k35_cifar10_' + 'd_'+ date + '_'+ 'e_v_'+ filename 
)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100, verbose=3, batch_size=100, validation_split=0.2,
          callbacks=[es, mcp])

#4. 평가 & 예측
results = model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])

'''
1/13 결과치
<<es, mcp 적용>>

313/313 [==============================] - 1s 3ms/step - loss: 2.3026 - acc: 0.1000
loss:  2.3026461601257324
acc:  0.10000000149011612

2. <<kernel_size, filters 값 변경 및 conv레이어 추가, batch_size= 128>>
loss:  3.044217109680176
acc:  0.48080000281333923

3. epoch= 10 , batch = 100
loss:  1.9259449243545532
acc:  0.4878000020980835

4.
loss:  1.842490792274475
acc:  0.4966000020503998

5.
Epoch 00026: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 3.0226 - acc: 0.4695
loss:  3.022554636001587
acc:  0.46950000524520874



1/25

1.  drop out, maxpooling, average pooling, padding 추가

313/313 [==============================] - 2s 4ms/step - loss: 0.9347 - acc: 0.6792
loss:  0.9346588850021362
acc:  0.6791999936103821


'''