import pandas as pd
import numpy as np

import datetime

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, concatenate
from sklearn.model_selection import train_test_split
from sklearn. preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import io

path = '/Users/hyunju/Desktop/study/_save/'

#삼성 주가 데이터와 아모레 주가 데이터 읽어오기, 시간 내림차순으로 정렬함.
samsung = pd.read_csv('/Users/hyunju/Desktop/study/samsung.csv', header=0, index_col=None, sep=',',
                      thousands=',').loc[::-1]

amore = pd.read_csv('/Users/hyunju/Desktop/study/amore.csv', header=0, index_col=None, sep=',',
                    thousands=',').loc[::-1]

# print(samsung)
# print(samsung.shape)      # (1980, 17)
# print(amore)
# print(amore.shape)        # (2220, 17)


# 삼성전자 주가의 x데이터와 y데이터 추출
sam_x = samsung[['고가', '저가', '종가', '외인(수량)', '개인', '기관']]
sam_y = samsung[['시가']]

# print(sam_x)
# print(sam_y)
# print(sam_x.shape)            # (1980, 6)
# print(sam_y.shape)            # (1980, 1)

# 아모레 주가의  x데이터와 y데이터 추출 
# 아모레 데이터가 2200행으로 삼성전자 데이터보다 많으므로 삼성전자와 같이 1980개의 행만 불러옴
amo_x = amore.loc[1979:0, ['고가', '저가', '종가', '외인(수량)', '개인', '시가']]
# print(amo_x)
print(amo_x.shape)            # (1980, 6)

scaler = MinMaxScaler()
sclaer2 = StandardScaler()

sam_x = scaler.fit_transform(sam_x)
amo_x = scaler.fit_transform(amo_x)

# split_x 함수를 통해 numpy array로 반환하기 위하여 데이터형 변환
sam_y = samsung[['시가']].to_numpy()

def split_x(dataset, timesteps):
    data = []
    for i in range(len(dataset) - timesteps + 1):
        sub_data = dataset[i: (i + timesteps)]
        data.append(sub_data)
    return np.array(data)

size=6

sam_x = split_x(sam_x, size)
amo_x = split_x(amo_x, size)

print(sam_x.shape)              # (1975, 6, 6)
print(amo_x.shape)              # (1975, 6, 6)


sam_y = sam_y[5:, :]              # 현재 삼성전자 시가 데이터 shape=(1980, 1)이므로 x데이터 크기와 맞추기 위해 5행을 제거함
print(sam_y.shape)              # (1975, 1)


#예측용 데이터 추출
sam_pred = sam_x[-1].reshape(-1, 6, 6)
amo_pred = amo_x[-1].reshape(-1, 6, 6)
print(sam_pred.shape)           # (1, 6, 6)
print(amo_pred.shape)           # (1, 6, 6)


sam_x_train, sam_x_test, amo_x_train, amo_x_test, sam_y_train, sam_y_test = train_test_split(
    sam_x, amo_x, sam_y, train_size=0.7, random_state=115)

# print(sam_x_train.shape, sam_x_test.shape)              # (1382, 6, 6) (593, 6, 6)
# print(amo_x_train.shape, sam_x_test.shape)              # (1382, 6, 6) (593, 6, 6)
# print(sam_y_train.shape, sam_y_test.shape)              # (1382, 1) (593, 1)



# 삼성전자
input1 = Input(shape=(6, 6))
dense1 = LSTM(128, return_sequences=True, activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = LSTM(64, activation='relu')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(64, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(32, activation='relu')(drop3)
dense5 = Dense(32, activation='relu')(dense4)
dense6 = Dense(16, activation='relu')(dense5)
drop4 = Dropout(0.2)(dense6)
dense7 = Dense(8, activation='relu')(drop4)
output1 = Dense(1)(dense7)

# 아모레
input2 = Input(shape=(6, 6))
dense11 = LSTM(128, return_sequences=True, activation='relu')(input2)
drop11 = Dropout(0.2)(dense11)
dense21 = LSTM(64, activation='relu')(drop11)
drop21 = Dropout(0.2)(dense2)
dense31 = Dense(64, activation='relu')(drop21)
drop31 = Dropout(0.2)(dense3)
dense41 = Dense(32, activation='relu')(drop31)
dense51 = Dense(32, activation='relu')(dense41)
dense61 = Dense(16, activation='relu')(dense51)
drop41 = Dropout(0.2)(dense61)
dense71 = Dense(8, activation='relu')(drop41)
output2 = Dense(1)(dense71)

# merge
merge1 = concatenate([output1, output2])
merge2 = Dense(128, activation='relu')(merge1)
merge3 = Dense(64, activation='relu')(merge2)
merge4 = Dense(64, activation='relu')(merge3)
merge5 = Dense(32, activation='relu')(merge4)
merge6 = Dense(32, activation='relu')(merge5)
merge7 = Dense(16, activation='relu')(merge6)
merge8 = Dense(16, activation='relu')(merge7)
merge9 = Dense(8, activation='relu')(merge8)
last_output = Dense(1, activation='relu')(merge9)


model = Model(inputs=[input1, input2], outputs=last_output)


#3. compile and training

#earlystopping 기준 설정
es = EarlyStopping(
    monitor='val_loss',  # history의 val_loss의 최소값을 이용함
    mode='min',  # max로 설정 시 갱신 안됨. (accuracy 사용 시에는 정확도 높을수록 좋기 때문에 max로 설정)
    patience=25,  # earlystopping n번 (최저점 나올 때까지 n번 돌림. 그 안에 안 나오면 종료)
    restore_best_weights=True,  # 이걸 설정해줘야 종료 시점이 아닌 early stopping 지점의 최적 weight 값 사용 가능
    verbose=1
)
model.compile(loss='mse', optimizer='adam', )

date = datetime.datetime.now()
print(date)
print(type(date))                           # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")

print(date)                                

# filepath = 'C:/study/_save/MCP/'
filepath = '/Users/hyunju/Desktop/study/_save/MCP/'
# d: digit, f: float
filename = '{epoch:04d}-{val_loss: .4f}.hdf5'


#ModelCheckpoint 설정
mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=1,
    save_best_only=True,                                              # save_best_only: 가중치 가장 좋은 지점 저장!
    # filepath= path + 'MCP/keras30_ModelCheckPoint3.hdf5'
    filepath= filepath + 'stock_' + 'd_'+ date + '_'+ 'e_v_'+ filename                      #파일명 날짜, 시간 넣어서 저장하기
)


model.fit([sam_x_train, sam_x_train], sam_y_train, epochs=1000, batch_size=80,
          validation_split=0.2, verbose=1, callbacks=[es, mcp]
          )

# model.save(path + 'keras51_conv1d_save_model_fetch.h5')

model.save_weights(path + 'stock_save_weight.h5') 


loss=model.evaluate([sam_x_test, amo_x_test], sam_y_test)
sam_y_pred=model.predict([sam_pred, amo_pred])

print("loss : ", loss)
print("삼성전자 예상 시가 :" , sam_y_pred)



'''
1/29일 

1.
loss :  18240014336.0
삼성전자 예상 시가 : [[163444.17]

2. 
loss :  25356886016.0
삼성전자 예상 시가 : [[179560.12]]

3.
loss :  3952441088.0
삼성전자 예상 시가 : [[70935.125]]

4.
loss :  27409698816.0
삼성전자 예상 시가 : [[213783.94]]

5. batch = 80
loss :  5185999360.0
삼성전자 예상 시가 : [[65311.336]]
'''

