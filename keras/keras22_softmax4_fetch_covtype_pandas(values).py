import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd


from sklearn.preprocessing import OneHotEncoder

#one-hot encoding 하는 방법: 판다스 사이킷런 텐서플로

#1. data
datasets = fetch_covtype()
x = datasets['data']
y = datasets['target']

# print(x.shape, y.shape)                             #(581012, 54) (581012,)         
# print(np.unique(y, return_counts=True))             #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
                                                      #    dtype=int64))
                                                                                                 


#2. 방법2 Pandas Getdummies

#y = pd.get_dummies(y, dummy_na=False)                   #dummy_na: True = (581012, 8),  //  False = (581012, 7)   (defalut값: false)  
                                                         # 마지막에 에러뜸  => data 형태: pandas.  np.argmax 부분에서 y_test가 pandas 형태여서
                                                         # numpy 자료형이 pandas를 바로 못 받아들이기 때문!!!!!
                                                         # (np.argmax  대신 tf.argmax 쓰면 에러 안 남)
                                                         #import pandas as pd

y = pd.get_dummies(y)
# print('y1 ',type(y))                                   # <class 'pandas.core.frame.DataFrame'>

y = y.values                                             #values 쓰거나 .numpy() 쓰면 오류 해결 =>y = pandas 데이터를 numpy 데이터로 바꿔주는 과정 (뒷쪽 에러 잡기 위해)

# print(type(y))                                         #<class 'numpy.ndarray'>
# print(datasets.head())
# print(y.shape)                                        #(581012, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=115, test_size=0.2, stratify=y)


#2. modeling
model = Sequential()
model.add(Dense(30, activation='linear',input_shape=(54, )))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(7, activation='softmax'))


#3. compile and training

from tensorflow.keras.callbacks import EarlyStopping                    #EarlyStopping 추가

#earlystopping 기준 설정
earlyStopping = EarlyStopping(
    monitor='val_loss',             #history의 val_loss의 최소값을 이용함
    mode='min',                     #max로 설정 시 갱신 안됨. (accuracy 사용 시에는 정확도 높을수록 좋기 때문에 max로 설정)
    patience=25,                     #earlystopping n번 (최저점 나올 때까지 n번 돌림. 그 안에 안 나오면 종료) 
    restore_best_weights=True,          #이걸 설정해줘야 종료 시점이 아닌 early stopping 지점의 최적 weight 값 사용 가능
    verbose=1
)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=1000, 
          validation_split=0.2, verbose =1, callbacks=[earlyStopping]
          )

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


loss:  0.41360634565353394
accuracy:  0.829582691192627




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


loss:  0.529424250125885
accuracy:  0.7766581177711487
acc:  0.7766580897222963
'''