import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. data

#불러올 데이터 경로
path = './_data/bike/'

#csv file 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission = pd.read_csv(path+'sampleSubmission.csv', index_col=0)

print(train_csv.columns)            #column =10

#결측치 처리: 삭제!
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.shape)      #(10886, 11)

x = train_csv.drop(['casual', 'registered','count'], axis=1)
#x = train_csv.drop('registered', axis=1)
y = train_csv['count']

print(x.shape)      #(10886, 8)
print(y.shape)      #(10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.7, random_state = 115)

print('x: ', x_train.shape, x_test.shape)          #(7620, 8) (3266, 8)
print('y: ' ,y_train.shape, y_test.shape)          #(7620,) (3266,)


#2. modeling
model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))            #제일 마지막값은 sigmoid 안돼~!! 마지막 값이 다 0~1로 바뀌니까!
#print(x.info())

#3. complile and training
import time
model.compile(loss='mse', optimizer='adam')
start = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size = 20)
end = time.time()


#4. Evaluation and Prediction
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
print('y_predict: ',y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print('RMSE: ', rmse)
print('time: ', end-start)

#제출용
y_submit = model.predict(test_csv)
print('y_sub: ', y_submit.shape)

r2 = r2_score(y_test, y_predict)

print('R2: ', r2)

#submission file 만들기
submission['count'] = y_submit
submission.to_csv(path+'submission_0106.csv')


#tf27 time: 890.0578079223633
'''
RMSE:  156.8333462819354
R2:  0.25922287444559156
'''
#tf274gpu time: 2221.257954120636 
'''
RMSE:  157.19649146826646
R2:  0.25578838685080696


3.
tf27, batch_size = 32

RMSE:  150.91685581156477
time:  296.1480174064636
y_sub:  (6493, 1)
R2:  0.31405982066589666


4. batch_size = 20
RMSE:  152.49454162957022
time:  267.13360047340393
y_sub:  (6493, 1)
R2:  0.2996432104055111


5.  히든레이어  dense 높임 ,epochs= 5000,  batch_size = 32

RMSE:  152.13308063988018
time:  981.4283418655396
y_sub:  (6493, 1)
R2:  0.302959416090981

=> -값 다시 뜸;;


6. optimizer='relu'로 다 바꿈, epochs= 500
RMSE:  150.42040482578014
time:  93.6973237991333
y_sub:  (6493, 1)
R2:  0.3185652891778844


7. loss = 'mse' => 'mae'로
RMSE:  152.9082329303241
time:  97.02840161323547
y_sub:  (6493, 1)
R2:  0.29583816264572316                => mse가 더 나음


8. epochs=10000, batch_size = 20

RMSE:  154.2915925488274
time:  2865.8218972682953
y_sub:  (6493, 1)
R2:  0.28303943480072336

'''