'''
[실습]
1. train_set 0.7이상
2. r2: 0.8이상, rmse 사용

'''
'''
import sklearn as sk
print(sk.__version__)
'''


from sklearn.datasets import load_boston        #교육용 데이터셋 불러오기
                                                #최초 ctrl+f5 누르면 로컬저장소에 데이터 저장됨. 그 후 실행 시 속도 빠름
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

                                                
#1. data
dataset = load_boston()             #boston 집값 데이터
x = dataset.data                    #house data
y = dataset.target                  #house price

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=115)

# print(x.shape)          #(506, 13)
# print(y.shape)          #(506, )

# print(dataset.feature_names)        #data column name
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
 
# print(dataset.DESCR)                #Describe details of column data


#2. modeling
model = Sequential()
model.add(Dense(1, input_dim=13))
model.add(Dense(5, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. compile training
#model = LinearRegression()
model.compile(loss='mse', optimizer='adam', metrics=['mse'] )
model.fit(x_train, y_train, epochs=5000, batch_size=30,
          validation_split=0.3)

#4. evaluation prediction
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)
RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('r2: ', r2)


'''
<결과>

1.
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. compile training
#model = LinearRegression()
model.compile(loss='mae', optimizer='adam', metrics=['mse'] )
model.fit(x_train, y_train, epochs=300, batch_size=15)

=> loss:  [3.5224502086639404, 29.3668155670166]
    r2:  0.6604768695548058
 
 
    
2.
model.fit(x_train, y_train, epochs=1000, batch_size=30)

loss:  [3.3175272941589355, 27.106103897094727]
r2:  0.6866139701738597   

3. loss = mae -> mse
loss:  [20.621477127075195, 20.621477127075195]
r2:  0.7615856839248921

4.
loss:  [20.548568725585938, 20.548568725585938]
r2:  0.7624286033058361

5. epochs = 5000
loss:  [19.88042640686035, 19.88042640686035]
r2:  0.7701533178992249



6. relu 적용, validation_set 설정, 
model.fit(x_train, y_train, epochs=5000, batch_size=30, validation_split=0.3)

loss:  [17.63306999206543, 17.63306999206543]
r2:  0.7961360338588911


7.

'''

