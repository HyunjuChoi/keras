import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator                     # 이미지를 데이터로 변경 => 증폭 가능

# 1. data

train_datagen = ImageDataGenerator(
    rescale=1./255,             # 255로 나눈다 => min_max(255-0) scaling
    horizontal_flip=True,       # 수평선을 기준으로 image 반전
    vertical_flip=True,         # 수직선을 기준으로 image 반전
    width_shift_range=0.1,      # 0.1만큼 이동
    height_shift_range=0.1,     # 0.1만큼 이동
    rotation_range=5,           # image 회전
    zoom_range=1.2,             # image 120% 확대
    shear_range=0.7,            # 임의 전단 변환 (shearing transformation) 범위
    fill_mode='nearest'         # 이미지 shift 시켰을 때 빈 공간 생기면 가까이에 있는 데이터로 채우기
)

test_datagen = ImageDataGenerator(
    rescale=1./255              # 테스트 데이터는 리스케일링만 한다! => why? 정확한 평가 위해!(데이터 증폭하면 신뢰성 떨어짐, 데이터 조작이나 마찬가지)
)

# x = (80+80, 150, 150, 1)          (normal+ad 데이터 개수, size, size, color)
# y = (80+80, )

xy_train= train_datagen.flow_from_directory(                 # 폴더 안에 있는 이미지 데이터 가져오기 (x,y 데이터 함께 있음)
            'C:/study/_data/brain/train/',
            target_size=(100, 100),                          # 모든 데이터 사이즈 통일
            batch_size=10,                                   # 훈련 시 5개씩 잘라서 훈련(fit에서 굳이 안 잘라도 됨)
            class_mode='binary',
            color_mode='grayscale',
            shuffle=True

    )   # Found 160 images belonging to 2 classes.           

xy_test= test_datagen.flow_from_directory(                 # 폴더 안에 있는 이미지 데이터 가져오기
            'C:/study/_data/brain/test/',
            target_size=(100, 100),                          # 모든 데이터 사이즈 통일
            batch_size=10,                                    # 훈련 시 5개씩 잘라서 훈련(fit에서 굳이 안 잘라도 됨)
            class_mode='binary',
            color_mode='grayscale',
            shuffle=True

    )   # Found 120 images belonging to 2 classes.


# 2. modeling

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LSTM, Conv1D

model = Sequential()
model.add(Conv2D(128, (2,2), activation='relu', input_shape=(100, 100, 1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))         # y 데이터 0,1이므로 sigomoid~!! 
# model.add(Dense(2, activation='softmax'))       # compile loss = 'sparse_categorical_crossentropy' (원핫인코딩 안 했을 경우)

# 3. compile and training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])  
hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=10,
                    validation_data=xy_test,                    # 검증 데이터는 위에서 나눠준 xy_test 그대로 넣어주면 됨
                    validation_steps=16
                    )               # steps_per_epochs와 validation_steps는 배치사이즈로 나눈 값이 최대, 그 이상으로 설정하면 에러 날 수 있음

accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 훈련의 마지막 값만 확인하기
print('loss: ', loss[-1])
print('val_loss: ', val_loss[-1])
print('accuracy: ', accuracy[-1])
print('val_acc: ', val_acc[-1])



# 그림 그리기

import matplotlib.pyplot as plt
line1, line2 = plt.plot(val_loss, 'r:^', val_acc, 'g:v')             #그림이 이거 말하는 거 맞나요... => 아닌듯요....
plt.show()
# plt.imshow(xy_train[115], 'Blues')
# plt.show
# plt.plot(x, y_predict, color='red')             #가중치 구해서 나온 예측값 그래프 그리기
# plt.show()

# 4. evaluation and prediction