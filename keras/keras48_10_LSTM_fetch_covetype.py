import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

path =  'C:/study/_save/'

#one-hot encoding 하는 방법: 판다스 사이킷런 텐서플로

#1. data
datasets = fetch_covtype()
x = datasets['data']
y = datasets['target']

# print('r_y: ')
# print(y)

# print(x.shape, y.shape)                             #(581012, 54) (581012,)         
# print(np.unique(y, return_counts=True))             #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
                                                      #    dtype=int64))
                                                      
                                                      
#방법1. keras의 to_categorical
                                              
from tensorflow.keras.utils import to_categorical           #array 데이터에 0값이 없으면 알아서 0추가해서 컬럼 수가 바뀜 ㅠ
y = to_categorical(y)

# print(y)
print('================================')
# print(y.shape)                                            #(581012,8)
# print(type(y))                                            #자료형 찍어보고 그에 맞는 내장함수로 데이터 다루기

# print(np.unique(y[:,0], return_counts=True))              #모든 행의 0번째 열의 데이터 정보를 확인해 본다!
#print(np.unique(y2, return_counts=True))


#y =  y[:, 1:]           #(0번째 열 삭제)
# print(y2)
# print(y2.shape)

y = np.delete(y, 0, axis=1)                                 #0번째 열 삭제   (axis=0이면 행, axis=1이면 열)
# print(y2.shape)                                           #(581012, 7)
# print(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=115, test_size=0.2, stratify=y)

# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, x_test.shape)                  # (464809, 54) (116203, 54)

x_train = x_train.reshape(464809, 9, 6)
x_test = x_test.reshape(116203, 9, 6)

# print('x_test: ', x_test)

#2. modeling
model = Sequential()
model.add(LSTM(30, activation='linear',input_shape=(9, 6)))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(7, activation='softmax'))

# print(model.summary())                  #Total params: 3,807

#2. modeling
# input1 = Input(shape=(54, ))
# dense1 = Dense(30, activation='linear')(input1)
# drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(20, activation='relu')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(30, activation='relu')(drop2)
# drop3 = Dropout(0.2)(dense3)
# dense4 = Dense(20, activation='relu')(drop3)
# dense5 = Dense(10, activation='linear')(dense4)
# output1 = Dense(7, activation='softmax')(dense5)

# model = Model(inputs = input1, outputs = output1)

# print(model.summary())              #Total params: 3,807

#3. compile and training

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint                    #EarlyStopping 추가

#earlystopping 기준 설정
es = EarlyStopping(
    monitor='val_loss',             #history의 val_loss의 최소값을 이용함
    mode='min',                     #max로 설정 시 갱신 안됨. (accuracy 사용 시에는 정확도 높을수록 좋기 때문에 max로 설정)
    patience=25,                     #earlystopping n번 (최저점 나올 때까지 n번 돌림. 그 안에 안 나오면 종료) 
    restore_best_weights=True,          #이걸 설정해줘야 종료 시점이 아닌 early stopping 지점의 최적 weight 값 사용 가능
    verbose=1
)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

import datetime
date = datetime.datetime.now()
print(date)
print(type(date))                           # <class 'datetime.datetime'>
date= date.strftime("%m%d_%H%M")

print(date)                                 # 0112_1502

filepath = 'C:/study/_save/MCP/'
filename = '{epoch:04d}-{val_loss: .4f}.hdf5'                       # d: digit, f: float 


#ModelCheckpoint 설정
# mcp = ModelCheckpoint(
#     monitor='val_loss', mode='auto', verbose=1,
#     save_best_only=True,                                              # save_best_only: 가중치 가장 좋은 지점 저장!
#     # filepath= path + 'MCP/keras30_ModelCheckPoint3.hdf5' 
#     filepath= filepath + 'k48_LSTM_fetch_' + 'd_'+ date + '_'+ 'e_v_'+ filename                      #파일명 날짜, 시간 넣어서 저장하기            
# )


model.fit(x_train, y_train, epochs=1000, batch_size=1000, 
          validation_split=0.2, verbose =1, callbacks=[es]
          )

# model.save(path + 'keras48_LSTM_save_model_fetch.h5')           

#4. evaluation and prediction
loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy: ', accuracy)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

# print('원래 y_pred: ',y_predict)

y_predict = np.argmax(y_predict, axis=1)                        #argmax: y_predict의 가장 큰 값 => 자리값 뽑아줌

# print('바뀐 y_pred: ',y_predict)

y_test = np.argmax(y_test, axis=1)
#Shape of passed values is (116203, 1), indices imply (116203, 7)

# print([y_test])
# print(y_test)
# print(y_predict)
# print(y_test.shape, y_predict.shape)

# print('y_pred 예측값: ',y_predict)                                                  
# print('y_test 원래 값: ',y_test)                                                   
acc = accuracy_score(y_test, y_predict)                                           #y_test: 정수형, y_predict는 실수형이라 error 남

print('acc: ', acc)


# loss:  0.41360634565353394
# accuracy:  0.829582691192627




#data download 받다가 오류났을때
#print(datasets.get_data_home())
    
    
'''
<tf.argmax>
loss:  1.4840083122253418
accuracy:  0.4610036015510559
acc:  0.46100358854762785

<to_categorical>
loss:  3.6996772289276123
accuracy:  0.3991979658603668
acc:  0.3991979553023588


<epochs=1000, batch_size=1000, patience = 25>

loss:  0.44903087615966797
accuracy:  0.8118637204170227
acc:  0.8118637212464395


1/11일

<<min max>>
Epoch 00297: early stopping
3632/3632 [==============================] - 2s 558us/step - loss: 0.3684 - accuracy: 0.8506
loss:  0.3683554530143738
accuracy:  0.8506492972373962
acc:  0.8506492947686376

<<standard>>
Epoch 00257: early stopping
3632/3632 [==============================] - 2s 569us/step - loss: 0.3639 - accuracy: 0.8537
loss:  0.3638581931591034
accuracy:  0.8536612391471863
acc:  0.8536612651996937



1/25
<< cnn >>
Epoch 00202: early stopping
3632/3632 [==============================] - 4s 970us/step - loss: 0.2703 - accuracy: 0.8919
loss:  0.27028241753578186
accuracy:  0.8918960690498352
acc:  0.8918960784144988


1/27 << LSTM >>
Epoch 00228: early stopping
3632/3632 [==============================] - 5s 1ms/step - loss: 0.2780 - accuracy: 0.8880
loss:  0.27797338366508484
accuracy:  0.8879977464675903
acc:  0.887997728113732
'''