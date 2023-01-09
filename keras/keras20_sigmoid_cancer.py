from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping                    #EarylyStopping 추가

#1. data
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets['data']
y= datasets['target']
# #print(x.shape, y.shape)         #(569, 30) (569,)

x_train, x_test, y_train, y_test= train_test_split(x, y, shuffle=True, random_state=333, test_size=0.2)

#2. modeling
model = Sequential()
model.add(Dense(50, activation='linear', input_shape=(30,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))                           #이진분류 activation = 'sigmoid'로 고정!!

#3. compile and training
model.compile(loss='binary_crossentropy', optimizer='adam',         #이진분류 무조건 loss='binary_crossentropy'
                metrics=['accuracy'])                               #정확도 판단 가능
                                                                    #metrics 추가 시 hist의 history에도 accuracy 정보 추가 됨


#earlystopping 기준 설정
earlyStopping = EarlyStopping(
    monitor='val_loss',                     #history의 val_loss의 최소값을 이용함
    mode='min',                             #max로 설정 시 갱신 안됨. (accuracy는 높을수록 좋기 때문에 max로 설정)
    patience=20,                            #earlystopping 5번 (최저점 나올 때까지 5번 돌림. 그 안에 안 나오면 종료) 
    restore_best_weights=True,
    verbose=1
)

hist = model.fit(x_train, y_train, epochs=10000, batch_size=16, validation_split=0.2,
          callbacks=[earlyStopping], verbose=1)
      
#4. evaluation and prediction

# loss= model.evaluate(x_test, y_test)
# print('loss, accuracy: ', loss)

loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ',loss)
print('accuracy: ', accuracy)

# results = model.predict()

y_predict = model.predict(x_test)                   #sigmoid 통과 후 값

# print(y_predict[:10])
# print(y_test[:10])

from sklearn.metrics import r2_score, accuracy_score                    #Accuracy Score 추가

# acc = accuracy_score(y_test, y_predict)                               #그냥 돌리면 y_test랑 y_predict 실수/정수형이라 에러남


#구글링해서 찾은 방법
y_pred_1d = y_predict.flatten()                                         # 차원 펴주기
y_pred_class = np.where(y_pred_1d > 0.5, 1 , 0)                         #0.5보다 크면 1, 작으면 0

acc = accuracy_score(y_test, y_pred_class)
print('accuracy score: ', acc)


'''
print('=================================')
print(hist.history)
'''