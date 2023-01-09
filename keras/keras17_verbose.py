from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)         #(506, 13) (506,)

x_train, x_test, y_train, y_test= train_test_split(x, y, shuffle=True, random_state=333, test_size=0.2)

#2. modeling
model = Sequential()
#model.add(Dense(5, input_dim=13))                   #input_dim은 행과 열일 때만 표현 가능함
model.add(Dense(5, input_shape=(13, )))             #(13, ), input_shape: 다차원일 때 input_dim 대신 사용!
                                                    #if (100, 10, 5)라면 (10,5)로 표현됨. 맨 앞의 100은 데이터 개수
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. compile and training
import time
model.compile(loss='mse', optimizer='adam')

start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=1, 
          validation_split=0.2, verbose=2)                  #verbose: animation effect, default = True
                                                            #[==============================], 진행표시줄 on/off
end = time.time()

#4. evalutaion and prediction
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('time: ', end-start)


'''
verbose=0: time:  4.200374364852905             진행표시줄, epochs, loss 출력X
verbose=1: time:  4.815029859542847             진행표시줄, epochs, loss 출력O
verbose=2: time:  2.3697915077209473            진행표시줄만 출력 X
verbose=3~: time:  2.524513006210327            epoch만 출력

=> verbose=0 일 때가 시간이 더 빠르다 (자원 낭비 감소)

'''