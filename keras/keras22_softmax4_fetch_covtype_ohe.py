import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#one-hot encoding 하는 방법: 판다스 사이킷런 텐서플로

#1. data
datasets = fetch_covtype()
x = datasets['data']
y = datasets['target']

# print(x.shape, y.shape)                             #(581012, 54) (581012,)         
# print(np.unique(y, return_counts=True))             #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
                                                      #    dtype=int64))
                                                      
                                                                                        
# print(y.shape)                                      #(581012, 7)
# print(y)


#방법3. 원핫인코딩 (데이터 전처리)              << fit - transform - toarray >>

from sklearn.preprocessing import OneHotEncoder

y = y.reshape(-1,1)                      # 2차원 변환 후 인코딩 해야 함! => 왜??? 원핫인코딩 차원이 2차원이니까
# print(y.shape)                         # (581012, 1)   

#y= y.reshape(581012, 1)                 # reshape할 때 데이터의 순서 바뀌지 않게 주의!!!!     <<31라인이랑 같은 방법>>

ohe = OneHotEncoder()                   
ohe.fit(y)
y_ohe = ohe.transform(y)                # 결과값: (58012, 7)로 바뀜  =>data type = scipy 형태 
#y_ohe = ohe.fit_transform(y)           # fit이랑 transform 한번에 적용도 가능

y_ohe = y_ohe.toarray()                 # 필요한 것이 array이므로 array 반환. 여기서 안해줄거면 33라인 ohe = OneHotEncoder(sparse=False)해줘야 함!

                                        #shape 맞추는 작업 해야 error 안남

# print(y)                              

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, shuffle=True, random_state=115, test_size=0.2, stratify=y)

#y = ohe.fit_transform(x_train)ㅋㅋ

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


1/11>

loss:  0.44774898886680603
accuracy:  0.8091873526573181
acc:  0.809187370377701

Epoch 00642: early stopping
3632/3632 [==============================] - 2s 593us/step - loss: 0.4201 - accuracy: 0.8229
loss:  0.42012685537338257
accuracy:  0.8228789567947388
acc:  0.8228789273943014
s
'''