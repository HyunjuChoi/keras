import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.callbacks import EarlyStopping    

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#1. data

#불러올 데이터 경로
path = 'C:/study/_data/bike/'

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

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print('x: ', x_train.shape, x_test.shape)          #(7620, 8) (3266, 8)
print('y: ' ,y_train.shape, y_test.shape)          #(7620,) (3266,)


#2. modeling
# model = Sequential()
# model.add(Dense(10, input_dim=8, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='relu'))            #제일 마지막값은 sigmoid 안돼~!! 마지막 값이 다 0~1로 바뀌니까!
#print(x.info())

#2. modeling
input1 = Input(shape=(8, ))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(10, activation='relu')(dense1)
dense3 = Dense(10, activation='relu')(dense2)  
dense4 = Dense(100, activation='relu')(dense3)  
dense5 = Dense(10, activation='relu')(dense4)  
dense6 = Dense(10, activation='relu')(dense5)  
output1 = Dense(1, activation='relu')(dense6)

model = Model(inputs = input1, outputs = output1)  

#3. complile and training
model.compile(loss='mse', optimizer='adam')
earlyStopping = EarlyStopping(
    monitor='loss',         #history의 val_loss의 최소값을 이용함
    mode='min',                 #max로 설정 시 갱신 안됨. (accuracy는 높을수록 좋기 때문에 max로 설정)
    patience=20,                  #earlystopping 5번 (최저점 나올 때까지 5번 돌림. 그 안에 안 나오면 종료) 
    restore_best_weights=True,
    verbose=1
)
model.fit(x_train, y_train, epochs=10000, batch_size = 32, validation_split=0.3, 
          callbacks=[earlyStopping])


#4. Evaluation and Prediction
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
#print('y_predict: ',y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print('RMSE: ', rmse)

#제출용
#위에서 data 스케일링 했기 때문에 제출용 데이터도 스케일링 해줘야 평가 결과치 비슷하게 뜸

test_csv = scaler.transform(test_csv)
y_submit = model.predict(test_csv)
#print('y_sub: ', y_submit.shape)

r2 = r2_score(y_test, y_predict)

print('R2: ', r2)

#submission file 만들기
submission['count'] = y_submit
submission.to_csv(path+'submission_0111_early_minmax_2.csv')


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


9. earlystpping, monitor = loss, patience= 10, batch_size = 100
Epoch 00107: early stopping
103/103 [==============================] - 0s 3ms/step - loss: 23973.9297
loss:  23973.9296875
RMSE:  154.8351744895272
y_sub:  (6493, 1)
R2:  0.2779787140088592

10. batch  = 50
Epoch 00104: early stopping
103/103 [==============================] - 0s 3ms/step - loss: 23949.2363
loss:  23949.236328125
RMSE:  154.7554140990421
y_sub:  (6493, 1)
R2:  0.2787223934349099

11. monitor= 'val_loss'
Epoch 00090: early stopping
103/103 [==============================] - 0s 3ms/step - loss: 23833.2539
loss:  23833.25390625
RMSE:  154.38024477727078
R2:  0.2822153016016823

12. batch_size = 32
Epoch 00073: early stopping
103/103 [==============================] - 0s 926us/step - loss: 23832.4414
loss:  23832.44140625
RMSE:  154.37757166793054
R2:  0.282240158413808

13. monitor = loss
Epoch 00158: early stopping
103/103 [==============================] - 0s 496us/step - loss: 23441.2637
loss:  23441.263671875
RMSE:  153.10540909963947
R2:  0.29402094913031374

14.
Epoch 00163: early stopping
103/103 [==============================] - 0s 940us/step - loss: 23799.2773
loss:  23799.27734375
RMSE:  154.27015085598055
R2:  0.28323869102455856


15. dense=30 -> 100
Epoch 00115: early stopping
103/103 [==============================] - 0s 473us/step - loss: 24760.2051
loss:  24760.205078125
RMSE:  157.3537548079537
R2:  0.2542985858239366



16. loss:  23866.171875
RMSE:  154.4868053025065
R2:  0.2812240619230224


17. validation_split = 0.3
Epoch 00212: early stopping
103/103 [==============================] - 0s 921us/step - loss: 23884.4336
loss:  23884.43359375
RMSE:  154.54590495972437
R2:  0.28067401446979945

18. patience = 20
Epoch 00177: early stopping
103/103 [==============================] - 0s 939us/step - loss: 23751.9141
loss:  23751.9140625
RMSE:  154.11658003553202
R2:  0.28466500498554037



1/11일

<<min max scaler>>
Epoch 00021: early stopping
103/103 [==============================] - 0s 503us/step - loss: 70276.4219
loss:  70276.421875
RMSE:  265.0970226437377
R2:  -1.1165105896156429

<<standard scaler>>
Epoch 00875: early stopping
103/103 [==============================] - 0s 537us/step - loss: 24062.0938
loss:  24062.09375
RMSE:  155.11961015809138
R2:  0.2753235390302812



<<standard, transform(test_csv)>>
Epoch 00461: early stopping
103/103 [==============================] - 0s 490us/step - loss: 23308.1543
loss:  23308.154296875
RMSE:  152.67009888632924
R2:  0.2980297298279583

<<minmax, transform(test_csv)>>
Epoch 00471: early stopping
103/103 [==============================] - 0s 1ms/step - loss: 23143.4297
loss:  23143.4296875
RMSE:  152.12965391894457
R2:  0.30299081671274186
'''