from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

path = 'C:/study/_save/'                                #path = './_save/'  or   '../_save/'
# model.save(path + 'keras29_3_save_model.h5')              #model.save('C:/study/_save/keras29_1_save_model.h5')


#그래프 한글 깨짐 방지
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'                   #폰트 저장된 경로에서 불러오기
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)


#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)         #(506, 13) (506,)

# print(np.min(x), np.max(x))         # 0.0 1.0

x_train, x_test, y_train, y_test= train_test_split(x, y, shuffle=True, random_state=115, test_size=0.2)


#Scaler 설정
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)               # x_train fit한 가중치 값 범위에 맞춰서 x_test 데이터 변환. 
                                                # fit 말고 transform만 하면 됨
                                                
#print(x_train.shape, x_test.shape)              # (404, 13) (102, 13)

x_train = x_train.reshape(404, 13, 1, 1)
x_test = x_test.reshape(102, 13, 1, 1)
# print(x_train.shape, x_test.shape)              # (404, 13, 1, 1) (102, 13, 1, 1)                                      
                                                                                      
#2. modeling 
model = Sequential()        
model.add(Conv2D(64, (2,1), input_shape=(13, 1, 1)))
model.add(Conv2D(32, (2,1), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='linear'))

model.summary()


#3. compile and training
model.compile(loss='mse', 
              optimizer='adam', 
              metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint                    #ModelCheckpoint 추가

#earlystopping 기준 설정
es = EarlyStopping(
    monitor='val_loss',                     # history의 val_loss의 최소값을 이용함
    mode='min',                             # max로 설정 시 갱신 안됨. (accuracy 사용 시에는 정확도 높을수록 좋기 때문에 max로 설정)
    patience=10,                            # earlystopping n번 (최저점 나올 때까지 n번 돌림. 그 안에 안 나오면 종료) 
    restore_best_weights=False,              # 이걸 설정해줘야 종료 시점이 아닌 early stopping 지점의 최적 weight 값 사용 가능
    verbose=1
)

import datetime
date = datetime.datetime.now()
print(date)
print(type(date))                           # <class 'datetime.datetime'>
date= date.strftime("%m%d_%H%M")

print(date)                                 # 0112_1502

filepath = 'C:/study/_save/MCP/'
filename = '{epoch:04d}-{val_loss: .4f}.hdf5'                       # d: digit, f: float 


#ModelCheckpoint 설정
mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=1,
    save_best_only=True,                                              # save_best_only: 가중치 가장 좋은 지점 저장!
    # filepath= path + 'MCP/keras30_ModelCheckPoint3.hdf5' 
    filepath= filepath + 'k39_boston_' + 'd_'+ date + '_'+ 'e_v_'+ filename                      #파일명 날짜, 시간 넣어서 저장하기            
)


model.fit(x_train, y_train, epochs=2000, batch_size=10, 
          validation_split=0.2,
          verbose=1, callbacks=[es, mcp])           #val_loss를 기준으로 최소값이 n번 이상 갱신 안되면 훈련 중지      


model.save(path + 'keras39_cnn_save_model_boston.h5')           

                   
#4. evaluation and prediction
print('======================= 1. 기본 출력 =======================')               #원래대로 훈련시킨 모델 출력
mse, mae = model.evaluate(x_test, y_test)
print('mse: ', mse)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R2: ',r2)



'''
"""
#5. 시각화
plt.figure(figsize=(9,6))           #그래프 사이즈 설정
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')                #maker=선 무늬
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()                          #격자무늬 추가
plt.xlabel('에포크')                 #축 이름 추가
plt.ylabel('손실')
plt.title('보스턴 손실 그래프')            #그래프 이름 추가
# plt.legend()                      #선 이름(label) 출력, 그래프 없는 지점에 자동 설정
plt.legend(loc='upper right')       #loc=' ': 원하는 위치에 설정 가능
plt.show()




결과치 
1.Epoch 00059: early stopping
4/4 [==============================] - 0s 0s/step - loss: 27.3670
loss:  27.36704444885254
===================================
===================================


2.patience = 15로 수정
Epoch 00152: early stopping
4/4 [==============================] - 0s 0s/step - loss: 99.3449
loss:  99.34488677978516
===================================
===================================


1/11

<<min-max scaler>>
Epoch 00047: early stopping
4/4 [==============================] - 0s 0s/step - loss: 25.9766 - mae: 2.9904
mse:  25.9765682220459
mae:  2.9904370307922363
===================================

<<스케일 적용 안했을때>>
Epoch 00050: early stopping
4/4 [==============================] - 0s 0s/step - loss: 27.0054 - mae: 3.6519
mse:  27.005395889282227
mae:  3.651909351348877

<<standard scaler>>
Epoch 00071: early stopping
4/4 [==============================] - 0s 4ms/step - loss: 18.3473 - mae: 2.9134
mse:  18.347291946411133
mae:  2.9133944511413574


<<x_train 데이터 min max scaling>>
4/4 [==============================] - 0s 0s/step - loss: 24.7787 - mae: 2.8700
mse:  24.77870750427246
mae:  2.869971990585327

<< standard >>
Epoch 00056: early stopping
4/4 [==============================] - 0s 0s/step - loss: 14.8283 - mae: 2.5487
mse:  14.82828426361084
mae:  2.5487279891967773

Epoch 00051: early stopping
4/4 [==============================] - 0s 0s/step - loss: 21.1994 - mae: 3.0084
mse:  21.199399948120117
mae:  3.0083703994750977



1/12 

<<restore_best_weights=False>> 했을때 => 기본출력이 체크포인트보다 나은 경우

======================= 1. 기본 출력 =======================
4/4 [==============================] - 0s 1ms/step - loss: 17.0725 - mae: 2.5187
mse:  17.072492599487305
R2:  0.8046304124917125
======================= 2. load_model 출력 =======================
4/4 [==============================] - 0s 1ms/step - loss: 17.0725 - mae: 2.5187
mse:  17.072492599487305
R2:  0.8046304124917125
======================= 3. ModelCheckPoint 출력 =======================
4/4 [==============================] - 0s 1ms/step - loss: 18.2326 - mae: 2.5415
mse:  18.232637405395508
R2:  0.7913542636235694

loss는 train 데이터로 훈련했고 평가,예측은 test데이터로 하기 때문에 이런 경우가 발생할 수 있다 (데이터가 다르기 때문에)
=> 따라서 직접 돌려보고 판단해야 한다.


'''