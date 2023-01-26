import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping

#1. data
dataset = np.array([1,2,3,4,5,6,7,8,9,10])          #(10, )

# y 데이터가 따로 없기 때문에 dataset에서 직접 x data와 y data로 나눈다
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9]])
y = np.array([4,5,6,7,8,9,10])

# print(x)

print(x.shape, y.shape)             # (7, 3) (7,)

#rnn의 특징인 "연결 순환 구조" 모델링을 위해 데이터 reshape (dnn과의 차이점!)
# x = x.reshape(7,3,1)                # 3개씩 묶은 전체 7개의 데이터 => 묶음 데이터 1개 => 1개씩 순차적 계산 
                                    # [[[1],[2],[3]], [[2],[3],[4]], ... ]           
                                    
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.7, random_state=115)

# print(x_train.shape, x_test.shape)                  # (4, 3) (3, 3)

#2. modeling
model = Sequential()
# model.add(SimpleRNN(64, input_shape=(3,1)))
model.add(Dense(64, input_shape=(3, )))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. compile, training

es = EarlyStopping(
    monitor='loss',
    mode = 'min',
    patience = 50,
    restore_best_weights=True,
    verbose=1
)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es])

#4. evaluation, prediction
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_pred = np.array([8,9,10]).reshape(1,3,1)
# y_pred = np.array([8,9,10]).reshape(3, )
# print(y_pred)

result = model.predict(y_pred)
print('[8,9,10]의 예측값: ', result)