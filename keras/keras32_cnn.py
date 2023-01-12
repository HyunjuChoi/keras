from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten           #2차원 이미지 데이터 다루기 위한 import
                                                                    # flatten: 차원 펼치기

model = Sequential()

model.add(Conv2D(filters=10, kernel_size=(2,2),          # kernel_size: 조각낼 이미지 사이즈, 
                                                        # filters = 10: 조각낸 이미지 수치(장 수) 10장으로 늘린다.
                 input_shape=(5, 5, 1)                  # 원본 이미지 사이즈: 5 X 5의 1장(흑백 이미지)
))

model.add(Conv2D(filters=5, kernel_size=(2,2)))          # filter 수는 직접 수행해보며 적절한 값 찾는다 (다음 레이어의 output? 노드 같은 개념)
model.add(Flatten())                                    # Dense 적용해 주기 위해 차원 펼치기 수행
model.add(Dense(10))
model.add(Dense(1))

model.summary()