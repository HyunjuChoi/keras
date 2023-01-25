import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#1. data

#데이터 경로
path = 'C:/study/_data/ddarung/'
path2= 'C:/study/_save/'

#csv file 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)            #index_col: data가 아닌 index column이므로 데이터에 추가되지 않도록 인덱스 명시
                                                                    #column data는 10개지만 마지막 column인 count는 빼야하므로 최종 input_dim = 9
test_csv = pd.read_csv(path+ 'test.csv', index_col=0)               #submission위한 y값 없는 data set
submission = pd.read_csv(path+'submission.csv', index_col=0)
#print(train_csv)
#print(train_csv.shape)
#sub_set = pd.read_csv()

#print(train_csv.columns)            #print column
'''
Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],s
      dtype='object')
'''
print(train_csv.info())             #non-null: 결측치           =>data가 없는 것, 비어있는 값에는 임의의 값을 넣어서 테스트도 가능. 아예 데이터 삭제하는 것도 방법.
'''
Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
      dtype='object')
'''
#print(test_csv.info())
'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1459 entries, 3 to 2179
Data columns (total 10 columns):
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   hour                    1459 non-null   int64
 1   hour_bef_temperature    1457 non-null   float64
 2   hour_bef_precipitation  1457 non-null   float64
 3   hour_bef_windspeed      1450 non-null   float64
 4   hour_bef_humidity       1457 non-null   float64
 5   hour_bef_visibility     1457 non-null   float64
 6   hour_bef_ozone          1383 non-null   float64
 7   hour_bef_pm10           1369 non-null   float64
 8   hour_bef_pm2.5          1342 non-null   float64
 9   count                   1459 non-null   float64
dtypes: float64(9), int64(1)
memory usage: 125.4 KB
None
'''
#print(train_csv.describe())
'''
None
              hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5        count
count  1459.000000           1457.000000             1457.000000         1450.000000        1457.000000          1457.000000     1383.000000    1369.000000     1342.000000  1459.000000
mean     11.493489             16.717433                0.031572            2.479034          52.231297          1405.216884        0.039149      57.168736       30.327124   108.563400
std       6.922790              5.239150                0.174917            1.378265          20.370387           583.131708        0.019509      31.771019       14.713252    82.631733
min       0.000000              3.100000                0.000000            0.000000           7.000000            78.000000        0.003000       9.000000        8.000000     1.000000
25%       5.500000             12.800000                0.000000            1.400000          36.000000           879.000000        0.025500      36.000000       20.000000    37.000000
50%      11.000000             16.600000                0.000000            2.300000          51.000000          1577.000000        0.039000      51.000000       26.000000    96.000000
75%      17.500000             20.100000                0.000000            3.400000          69.000000          1994.000000        0.052000      69.000000       37.000000   150.000000
max      23.000000             30.000000                1.000000            8.000000          99.000000          2000.000000        0.125000     269.000000       90.000000   431.000000
'''

###### 결측치 처리 방법 1. 삭제 ######
#print(train_csv.isnull().sum())               #train_csv의 null값 count  //.info()와의 차이: null값을 보여주는지 null값 뺀 나머지를 보여주는지!!
train_csv = train_csv.dropna()                #결측치 제거
#print(train_csv.isnull().sum())
#print(train_csv.shape)

x = train_csv.drop('count', axis=1)         #delete count column
#print(x)                                    #[1458 rows x 9 columns]
y = train_csv['count']                      #train_csv에서 count column만 뽑아오기
# print(y)
# print(y.shape)                              #(1328, )

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.7, random_state=115)

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print('x: ', x_train.shape, x_test.shape)          #(929, 9) (399, 9)
# print('y: ' ,y_train.shape, y_test.shape)          #(929,) (399,)

x_train = x_train.reshape(929, 3, 3, 1)
x_test = x_test.reshape(399, 3, 3, 1)

# #2. modeling
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(3,3,1), padding='same'))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#2. modeling
# input1 = Input(shape=(9,))
# dense1 = Dense(10)(input1)
# drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(10, activation='relu')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(30, activation='relu')(drop2)
# drop3 = Dropout(0.2)(dense3)
# dense4 = Dense(50, activation='relu')(drop3)
# dense5 = Dense(100, activation='relu')(dense4)
# dense6 = Dense(50, activation='relu')(dense5)
# dense7 = Dense(30, activation='relu')(dense6)
# dense8 = Dense(10, activation='relu')(dense7)
# output1 = Dense(1)(dense8)

# model = Model(inputs = input1, outputs = output1)

#3. compile and training                                                       #컴파일 시간 재기
model.compile(loss='mse', optimizer='adam')#, metrics=['mse'])
es = EarlyStopping(
    monitor='val_loss',
    mode = min,
    patience = 50,
    restore_best_weights=True,
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
    # filepath= path2 + 'MCP/keras30_ModelCheckPoint3.hdf5' 
    filepath= filepath + 'k39_ddarung_' + 'd_'+ date + '_'+ 'e_v_'+ filename                      #파일명 날짜, 시간 넣어서 저장하기            
)


model.fit(x_train, y_train, epochs=1, batch_size=10, validation_split=0.3,
          callbacks=[es, mcp])

model.save(path2 + 'keras39_cnn_save_model_drop_ddarung.h5')    

#4. evaluation prediction
loss = model.evaluate(x_test,y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)               #=> 그냥 돌리면 결측치 때문에 nan 값 에러남
#print(y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))               #평가지표 확인

rmse = RMSE(y_test, y_predict)                                          #추정 값 또는 모델이 예측한 값과 실제 환경에서 관찰되는 값의 차이를 다룰 때 흔히 사용하는 측도
print('RMSE: ',rmse)


#제출용 
#위에서 data 스케일링 했기 때문에 제출용 데이터도 스케일링 해줘야 평가 결과치 비슷하게 뜸

test_csv = scaler.transform(test_csv)
# print(test_csv.shape)
test_csv = test_csv.reshape(715, 3, 3, 1)

y_submit = model.predict(test_csv)

#print(y_submit.shape)                   #(715, 1)
r2 = r2_score(y_test, y_predict)

print('R2: ',r2)

#submission2 = pd.DataFrame(y_submit)

#print('sub: ', submission2)


#결측치 수정하자~~!


#.to_csv()를 사용하여
#submission_0105.csv를 완성하자
#print(submission)
submission['count'] = y_submit          #비어있던 submission파일의 'count' 컬럼에 예측한 y_submit 값을 넣는다.
#print(submission)
submission.to_csv(path+'submission_0125_early_minmax_cnn.csv')




'''
결과치
1.model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#3. compile and training
model.compile(loss='mse', optimizer='adam')#, metrics=['mse'])
model.fit(x_train, y_train, epochs=500, batch_size=32)

RMSE:  53.40981815970996
R2:  0.5792660633745889


2. batch_size = 10
RMSE:  51.59100628978233
R2:  0.6074334032449771

3.
RMSE:  53.429657947849286
R2:  0.5789534308734482

4. 히든레이어 추가, epochs = 1000

RMSE:  52.05415124245194
R2:  0.600353435215272

5. 히든레이어 줄임, batch_size = 1
RMSE:  51.941006948473145
R2:  0.6020888812833705


6. ealry stopping, patience = 20 (ealrystopping 안걸림)
RMSE:  51.25642835390912
R2:  0.6125086376579238

7. patience =10 
Epoch 00066: early stopping
13/13 [==============================] - 0s 3ms/step - loss: 2578.7429
loss:  2578.742919921875
RMSE:  50.78132825041226
R2:  0.6196587254092925

8.
Epoch 00030: early stopping
13/13 [==============================] - 0s 2ms/step - loss: 2581.0862
loss:  2581.086181640625
RMSE:  50.804386922728526
R2:  0.6193132379520001

9. epochs = 1000
Epoch 00037: early stopping
13/13 [==============================] - 0s 2ms/step - loss: 2669.7976
loss:  2669.797607421875
RMSE:  51.670085843924724
R2:  0.6062290156568856


10. monitor = val_loss
Epoch 00036: early stopping
13/13 [==============================] - 0s 3ms/step - loss: 2519.3828
loss:  2519.3828125
RMSE:  50.193453078868615
R2:  0.6284138712523525


1/11일

<<min max scaler>>
Epoch 00084: early stopping
13/13 [==============================] - 0s 1ms/step - loss: 1740.8335
loss:  1740.83349609375
RMSE:  41.72329740791359
R2:  0.7432428228709054


<<standard scaler>>
Epoch 00043: early stopping
13/13 [==============================] - 0s 666us/step - loss: 2133.1428
loss:  2133.142822265625
RMSE:  46.185956170799585
R2:  0.6853807993530914

<<min max 2번째>>
Epoch 00056: early stopping
13/13 [==============================] - 0s 728us/step - loss: 1782.5774
loss:  1782.577392578125
RMSE:  42.220579874114975
R2:  0.7370859878450966

<<standard 2번째>>
Epoch 00069: early stopping
13/13 [==============================] - 0s 749us/step - loss: 1604.5242
loss:  1604.524169921875
RMSE:  40.0565122578922
R2:  0.7633472204006669


<<scaling 미적용>>
Epoch 00032: early stopping
13/13 [==============================] - 0s 727us/step - loss: 2888.1560
loss:  2888.156005859375
RMSE:  53.74156424226139
R2:  0.5740231953368953


<<standard, test_csv도 transform>>
Epoch 00362: early stopping
103/103 [==============================] - 0s 997us/step - loss: 22782.0156
loss:  22782.015625
RMSE:  150.9371365297972
R2:  0.3138754503494816


<<min max, test_csv도 transform>>
Epoch 00033: early stopping
13/13 [==============================] - 0s 667us/step - loss: 1963.5798
loss:  1963.579833984375
RMSE:  44.31229821553896
R2:  0.7103897724123318

<<standard, test_csv도 transform>>
Epoch 00060: early stopping
13/13 [==============================] - 0s 829us/step - loss: 1926.2952
loss:  1926.295166015625
RMSE:  43.88957698771199
R2:  0.7158889433674684


<<patience = 15, hidden layer 줄임, minmax>>
1.
Epoch 00086: early stopping
13/13 [==============================] - 0s 665us/step - loss: 1969.4463
loss:  1969.4462890625
RMSE:  44.37844341841008
R2:  0.709524521662312

2.
Epoch 00160: early stopping
13/13 [==============================] - 0s 585us/step - loss: 1592.8207
loss:  1592.8206787109375
RMSE:  39.91015497538985
R2:  0.7650734107273433

3.
Epoch 00151: early stopping
13/13 [==============================] - 0s 635us/step - loss: 1612.2905
loss:  1612.29052734375
RMSE:  40.15333959392197
R2:  0.7622017310849072

4.patience 50
Epoch 00153: early stopping
13/13 [==============================] - 0s 631us/step - loss: 1591.3390
loss:  1591.3389892578125
RMSE:  39.89158989200553
R2:  0.765291922397566




1/25

<<cnn>>
1.
loss:  1901.73828125
RMSE:  43.60892740195224
'''
