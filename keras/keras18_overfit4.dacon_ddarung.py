import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#1. data

#데이터 경로
path = './_data/ddarung/'

#csv file 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)            #index_col: data가 아닌 index column이므로 데이터에 추가되지 않도록 인덱스 명시
                                                                    #column data는 10개지만 마지막 column인 count는 빼야하므로 최종 input_dim = 9
test_csv = pd.read_csv(path+ 'test.csv', index_col=0)               #submission위한 y값 없는 data set
submission = pd.read_csv(path+'submission.csv', index_col=0)
#print(train_csv)
#print(train_csv.shape)
#sub_set = pd.read_csv()

print(train_csv.columns)            #print column
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
print(test_csv.info())
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
print(train_csv.describe())
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
print(train_csv.isnull().sum())               #train_csv의 null값 count  //.info()와의 차이: null값을 보여주는지 null값 뺀 나머지를 보여주는지!!
train_csv = train_csv.dropna()                #결측치 제거
print(train_csv.isnull().sum())
print(train_csv.shape)

x = train_csv.drop('count', axis=1)         #delete count column
print(x)                                    #[1458 rows x 9 columns]
y = train_csv['count']                      #train_csv에서 count column만 뽑아오기
print(y)
print(y.shape)                              #(1328, )

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.7, random_state=115)

print('x: ', x_train.shape, x_test.shape)          #(929, 9) (399, 9)
print('y: ' ,y_train.shape, y_test.shape)          #(929,) (399,)

#2. modeling
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(10, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. compile and training
                                              
model.compile(loss='mse', optimizer='adam')#, metrics=['mse'])
hist = model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.3)


#4. evaluation prediction
loss = model.evaluate(x_test,y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)               #=> 그냥 돌리면 결측치 때문에 nan 값 에러남
print(y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))               #평가지표 확인

rmse = RMSE(y_test, y_predict)                                          #추정 값 또는 모델이 예측한 값과 실제 환경에서 관찰되는 값의 차이를 다룰 때 흔히 사용하는 측도
print('RMSE: ',rmse)


#제출용 
y_submit = model.predict(test_csv)

print(y_submit.shape)                   #(715, 1)
r2 = r2_score(y_test, y_predict)

print('R2: ',r2)

#submission2 = pd.DataFrame(y_submit)

#print('sub: ', submission2)


#결측치 수정하자~~!


#.to_csv()를 사용하여
#submission_0105.csv를 완성하자
print(submission)
submission['count'] = y_submit          #비어있던 submission파일의 'count' 컬럼에 예측한 y_submit 값을 넣는다.
print(submission)
submission.to_csv(path+'submission_0109_val.csv')


#5. 시각화
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('dacon_ddarung loss')
plt.legend()
plt.show()