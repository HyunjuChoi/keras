import numpy as np
from sklearn.datasets import load_wine                          #와인 감정 데이터
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf

#1. data
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)                           #(178, 13) (178,)
# print(y)
# print(np.unique(y))                               #라벨의 unique값(데이터 값 종류) 출력
                                                    #[0 1 2]      (데이터 값 종류(class) 많을 때 유용)

# print(np.unique(y, return_counts=True))           #각 class의 개수 출력, (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

# print(datasets.feature_names)                       #['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 
                                                      #   'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
                                                    
from tensorflow.keras.utils import to_categorical      #one-hot encoding => [[1. 0. 0.] [1. 0. 0.] [1. 0. 0.] ...] 이런 식으로 변환 ( ex, [1. 0. 0.] = 0)
y = to_categorical(y)

# print(y)
# print(y.shape)                      #(178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=115, test_size=0.2, stratify=y)

# print('x_test: ', x_test)

#2. modeling
model = Sequential()
model.add(Dense(30, activation='linear',input_shape=(13, )))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(3, activation='softmax'))


#3. compile and training

from tensorflow.keras.callbacks import EarlyStopping                    #EarlyStopping 추가

#earlystopping 기준 설정
earlyStopping = EarlyStopping(
    monitor='val_loss',             #history의 val_loss의 최소값을 이용함
    mode='min',                     #max로 설정 시 갱신 안됨. (accuracy 사용 시에는 정확도 높을수록 좋기 때문에 max로 설정)
    patience=15,                     #earlystopping n번 (최저점 나올 때까지 n번 돌림. 그 안에 안 나오면 종료) 
    restore_best_weights=True,          #이걸 설정해줘야 종료 시점이 아닌 early stopping 지점의 최적 weight 값 사용 가능
    verbose=1
)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=1, 
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
# print('y_pred 예측값: ',y_predict)                                                  
# print('y_test 원래 값: ',y_test)                                                   
acc = accuracy_score(y_test, y_predict)                                           #y_test: 정수형, y_predict는 실수형이라 error 남

print('acc: ', acc)