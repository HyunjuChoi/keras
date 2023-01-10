import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd

import scipy.sparse


from sklearn.preprocessing import OneHotEncoder

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
                                                      
                                                      
                                              
# from tensorflow.keras.utils import to_categorical           #array 데이터에 0값이 없으면 알아서 0추가해서 컬럼 수가 바뀜 ㅠ
# y = to_categorical(y)

# print('y1: ')
print(y)
print('================================')
# print(y.shape)
# print(type(y))

#y =  y[:, 1:]           #(0번째 열 삭제)
# print(y2)
# print(y2.shape)

# print(np.unique(y, return_counts=True))
# print(np.unique(y2, return_counts=True))

#y = np.delete(y, 0, axis=1)                 #0번째 열 삭제
# print(y2.shape)
#print(y)




#y = pd.get_dummies(y, dummy_na=False)                   #dummy_na: True = (581012, 8),  //  False = (581012, 7)   (defalut값: false)  
                                                         # 마지막에 에러뜸  =>np.argmax  대신 tf.argmax 쓰면 에러 안 남ㄴ
                                                         #import pandas as pd
y = pd.get_dummies(y)
#print(type(y))
y = y.values           #.numpy()


# print(datasets.head())
# print(y.shape)                      #(581012, 7)
# print(y)

#y.cat.remove_unused_categories()
'''
y = y.reshape(-1,1)

ohe = OneHotEncoder()                   #???????????????????
ohe.fit(y)
y_ohe = ohe.transform(y)
y_ohe = y_ohe.toarray()
#shape 맞추는 작업 해야 error 안남


'''

# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=115, test_size=0.2, stratify=y)

#y = ohe.fit_transform(x_train)

# print('x_test: ', x_test)

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
model.fit(x_train, y_train, epochs=1, batch_size=1000, 
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
'''