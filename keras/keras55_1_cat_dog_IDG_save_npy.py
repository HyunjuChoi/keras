import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator                     # 이미지를 데이터로 변경 => 증폭 가능

train_datagen = ImageDataGenerator(
    rescale=1./255,             # 255로 나눈다 => min_max scaling
    # horizontal_flip=True,       # 수평선을 기준으로 image 반전
    # vertical_flip=True,         # 수직선을 기준으로 image 반전
    # width_shift_range=0.1,      # 0.1만큼 이동
    # height_shift_range=0.1,     # 0.1만큼 이동
    # rotation_range=5,           # image 회전
    # zoom_range=1.2,             # image 120% 확대
    # shear_range=0.7,            # 임의 전단 변환 (shearing transformation) 범위
    # fill_mode='nearest'         # 이미지 shift 시켰을 때 빈 공간 생기면 가까이에 있는 데이터로 채우기
)

test_datagen = ImageDataGenerator(
    rescale=1./255              # 테스트 데이터는 리스케일링만 한다! => why? 정확한 평가 위해!(데이터 증폭하면 신뢰성 떨어짐, 데이터 조작이나 마찬가지)
)

# x = (80+80, 150, 150, 1)          (normal+ad 데이터 개수, size, size, color)
# y = (80+80, )

xy_train= train_datagen.flow_from_directory(                 # 폴더 안에 있는 이미지 데이터 가져오기
            'D:/_data/DC/train/train/',
            target_size=(200, 200),                          # 모든 데이터 사이즈 통일
            batch_size=1,                                
            class_mode='binary',
            # class_mode='categorical',
            color_mode='rgb',
            shuffle=True

    )   # Found 25000 images belonging to 2 classes.          

xy_valid= train_datagen.flow_from_directory(                 # 폴더 안에 있는 이미지 데이터 가져오기
            'D:/_data/DC/train/train/',
            target_size=(200, 200),                          # 모든 데이터 사이즈 통일
            batch_size=1000,                                
            class_mode='binary',
            # class_mode='categorical',
            color_mode='rgb',
            shuffle=True,
            # subset='validation'

    )   # Found 25000 images belonging to 2 classes.

xy_test= test_datagen.flow_from_directory(                 # 폴더 안에 있는 이미지 데이터 가져오기
            'D:/_data/DC/test1/',
            target_size=(200, 200),                          # 모든 데이터 사이즈 통일
            batch_size=1000,                                
            class_mode='binary',
            # class_mode='categorical',                       # one hot encoding 되어서 나옴
            color_mode='rgb',
            shuffle=True

    )   # Found 12500 images belonging to 1 classes.

# print(xy_train.shape)    
print(xy_test[0][1].shape)          # (1000, 200, 200, 3)   

# 먼소리여..... 
print(xy_train[0][0].shape)         # (1000, 200, 200, 3)                 => 배치사이즈 1000으로 나눴으니까 25개 행에 (1000, 200, 200, 1)씩 들어가 있음. 
print(xy_train[0][1].shape)         # (1000, )                            
'''
x data: xy_train[0][0] ~ [24][0]까지
y data: xy_train[0][1] ~ [24][1]까지
'''

### 이미지를 수치화 할 때 시간이 오래 걸리므로 numpy 파일로 저장하여 불러온다 (시간 단축) ### 

np.save('C:/study/_data/DC/dc_x_train.npy', arr=xy_train[0][0])                   # numpy 파일 생성하여 x_train 데이터 저장
np.save('C:/study/_data/DC/dc_y_train.npy', arr=xy_train[0][1])                   # numpy 파일 생성하여 y_train 데이터 저장

# np.save('C:/study/_data/brain/brain_train.npy', arr=xy_train[0])                      # numpy 파일 생성하여 train 데이터 저장 => 안됨!
                                                                                        # numpy는 형태가 맞아야 하는데 얘는 튜플이라 다름.. 그래서 안됨. 
                                                                                        # => append로 와꾸 맞춘 후 저장
                                                                                        # 추후에 x,y 나눠줘야 함

np.save('C:/study/_data/DC/dc_x_valid.npy', arr=xy_valid[0][0])                     # numpy 파일 생성하여 x_valid 데이터 저장
np.save('C:/study/_data/DC/dc_y_valid.npy', arr=xy_valid[0][1])                     # numpy 파일 생성하여 y_valid 데이터 저장

# np.save('C:/study/_data/DC/dc_test.npy', arr=xy_test)                     # numpy 파일 생성하여 xy_test 데이터 저장
# np.save('C:/study/_data/DC/dc_y_test.npy', arr=xy_valid[0][1])                     # numpy 파일 생성하여 y_valid 데이터 저장


print(type(xy_train))               # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))            # <class 'tuple'>          리스트와의 차이점: 생성 후 데이터 변경(수정) 불가
print(type(xy_train[0][0]))         # <class 'numpy.ndarray'>
print(type(xy_train[0][1]))         # <class 'numpy.ndarray'>