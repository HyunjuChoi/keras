import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=1000, batch_size = 500, validation_split=0.2, verbose=1)



#4. Evaluation and Prediction
loss= model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
print('y_predict: ',y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print('RMSE: ', rmse)

#제출용
y_submit = model.predict(test_csv)
print('y_sub: ', y_submit.shape)

r2 = r2_score(y_test, y_predict)

print('R2: ', r2)

#submission file 만들기
submission['count'] = y_submit
submission.to_csv(path+'submission_0109.csv')



#5. 시각화
# plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Kaggle Bike Loss')
plt.legend()
plt.show()





