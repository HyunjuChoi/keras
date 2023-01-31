import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator                     # 이미지를 데이터로 변경 => 증폭 가능

### 이미지를 수치화 할 때 시간이 오래걸리므로 numpy 파일로 저장하여 불러온다 (시간 단축) ### 

# np.save('C:/study/_data/brain/brain_x_train.npy', arr=xy_train[0][0])                   # numpy 파일 생성하여 x_train 데이터 저장
# np.save('C:/study/_data/brain/brain_y_train.npy', arr=xy_train[0][1])                   # numpy 파일 생성하여 y_train 데이터 저장
# # np.save('C:/study/_data/brain/brain_train.npy', arr=xy_train)                         # numpy 파일 생성하여 train 데이터 저장 => 추후에 x,y 나눠줘야 함

# np.save('C:/study/_data/brain/brain_x_test.npy', arr=xy_test[0][0])                     # numpy 파일 생성하여 x_test 데이터 저장
# np.save('C:/study/_data/brain/brain_y_test.npy', arr=xy_test[0][1])                     # numpy 파일 생성하여 y_test 데이터 저장


### 저장한 numpy 파일 불러오기 ###
x_train = np.load('C:/study/_data/brain/brain_x_train.npy')
y_train = np.load('C:/study/_data/brain/brain_y_train.npy')
x_test = np.load('C:/study/_data/brain/brain_x_test.npy')
y_test = np.load('C:/study/_data/brain/brain_y_test.npy')

print(x_train.shape, x_test.shape)                  # (160, 200, 200, 1) (120, 200, 200, 1)
print(y_train.shape, y_test.shape)                  # (160,) (120,)


# 2. modeling

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), activation='relu', input_shape=(200, 200, 1)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(Conv2D(16, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))         # y 데이터 0,1이므로 sigomoid~!! 

# 3. compile and training

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience = 25,
    restore_best_weights=True,
)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=1000,                       # xy_train[0][0]: x data, xy_train[0][1]: y data
                                                                                    # batch_size 통으로 잘라서 160개 통채로 들어가 있으므로 [0][0], [0][1]이 가능함
                    #validation_data=xy_test,
                    batch_size= 32,                                 # batch_size에 왜 영향 받는지?????              
                    #validation_steps=16
                    validation_data=(x_test, y_test),
                    callbacks=[es]
                    )    

# hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=100,
#                     validation_data=xy_test,                    # 검증 데이터는 위에서 나눠준 xy_test 그대로 넣어주면 됨
#                     validation_steps=16
#                     )               # steps_per_epochs와 validation_steps는 배치사이즈로 나눈 값이 최대, 그 이상으로 설정하면 에러 날 수 있음

accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 훈련의 마지막 값만 확인하기
print('loss: ', loss[-1])
print('val_loss: ', val_loss[-1])
print('accuracy: ', accuracy[-1])
print('val_acc: ', val_acc[-1])