import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# 모델구성칸에 넣어도 되는데 가독성을 위해 import하는 것들은 위에 적음.
# Sequential 하기 전에만 이것을 적어두면 됨.

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

#2. 모델구성
model = Sequential()
# Sequential을 model이란 이름으로 정의함. 모델 이름 = 모델 종류()
model.add(Dense(3,input_dim=1))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))
# add란 model이란 이름의 모델 안에 추가한다는 의미다.
# dim은 dimension으로 차원을 의미한다. 방향은 (y:출력, x:입력)이다.
# 심층 신경망 1-3-5-4-2-1 구성한 것임. 각각 코드 한 줄이 레이어이고 dense 안에 있는 숫자가 노드이다.
# hidden layer에 있는 input_dim=숫자는 앞에 있는 것과 연결되기 때문에 삭제해도 된다.

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=200)

#4. 평가, 예측
result = model.predict([6])
print('6의 결과 : ', result)
# 결과값이 잘 나온 경우는 초기의 랜덤값이 잘 설정된 것이다.
# 결과값이 잘 나오게 하는 방법은 훈련 횟수(epochs)를 늘리거나 신경망의 중간 계층 수(hidden layer)를 많이 구성하는 것이다.

# 파라미터(Parameter) : 머신러닝 훈련 모델에 의해 요구되는 매개 변수
# 하이퍼 파라미터(Hyperparameter) : 최적의 훈련 모델의 구현하기 위해 모델에 설정하는 변수
# 하이퍼 파라미터 : 1 학습률(Learning Rate) 2 훈련 반복 횟수 (epochs) 3 가중치 초기화
# 하이퍼 파라미터의 튜닝 기법 : 그리드 탐색, 랜덤 탐색, 베이지안 최적화

"""
결과
Epoch 200/200
1/1 [==============================] - 0s 997us/step - loss: 0.4040
6의 결과 :  [[5.8864875]]
"""