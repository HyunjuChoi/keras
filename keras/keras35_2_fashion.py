from tensorflow.keras.datasets import fashion_mnist

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()                  #텐서플로 mnist 데이터셋 불러와서 변수에 저장하기

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, AveragePooling2D, Dropout

model = Sequential()
model.add(Conv2D(filters=128, kernel_size = (2,2), input_shape=(28, 28, 1), 
                 padding='same',
                 strides=1,
                 activation='relu'))                            # (28, 28, 128), Conv2D의 default stride = 1

model.add(MaxPooling2D())                                       # Max Pooling의 default stride 값 = 2
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size = (2,2), padding='same')) 
model.add(MaxPooling2D())                                       # (14, 14, 128)
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size = (2,2), padding='same'))       
model.add(MaxPooling2D())                                       # (14, 14, 128)
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size = (2,2), padding='same'))       
model.add(MaxPooling2D())                                       # (14, 14, 128)
model.add(Dropout(0.25))                 
model.add(Conv2D(filters=16, kernel_size = (3,3), padding='same'))           
model.add(Flatten())                                            # 25*25*32 = 40,000
model.add(Dense(32, activation='relu'))                         # input_shape = (60000, 40000)에서 '행 무시'이므로 (40000, )
                                                                # 60000 = batch_size, 40000 = input_dim
                                                                
model.add(Dense(10, activation='softmax'))                      #output 노드 10개이므로 다중분류! 

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
    filepath= filepath + 'k35_fashion_' + 'd_'+ date + '_'+ 'e_v_'+ filename 
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

1. << max pooling, average pooling, padding 추가>>

Epoch 00036: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.3052 - acc: 0.9021
loss:  0.30520984530448914
acc:  0.9021000266075134
Accuracy: 90.21%

2.  conv2D 레이어 여러개 추가, dropout 추가
313/313 [==============================] - 1s 3ms/step - loss: 0.2687 - acc: 0.9019
loss:  0.26868292689323425
acc:  0.9018999934196472
Accuracy: 90.19%

'''