import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator                     # 이미지를 데이터로 변경 => 증폭 가능 (실시간 이미지 증가(augumentation))

train_datagen = ImageDataGenerator(
    rescale=1./255,             # 255로 나눈다 => min_max scaling
    
    #### 여기서 데이터 변경 이루어지면 그만큼 데이터가 배로 생성됨을 유의 ####
    horizontal_flip=True,       # 수평선을 기준으로 50% 확률로 image 반전
    vertical_flip=True,         # 수직선을 기준으로 50% 확률로 image 반전
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

xy_train= train_datagen.flow_from_directory(                 # 폴더 안에 있는 이미지 데이터 가져오기
            'C:/study/_data/brain/train/',
            target_size=(200, 200),                          # 모든 데이터 사이즈 통일
            batch_size=10,                                   # 훈련 시 10개씩 잘라서 훈련(fit에서 굳이 안 잘라도 됨)
            class_mode='binary',
            color_mode='grayscale',
            shuffle=True

    )   # Found 160 images belonging to 2 classes.           

xy_test= test_datagen.flow_from_directory(                 # 폴더 안에 있는 이미지 데이터 가져오기
            'C:/study/_data/brain/test/',
            target_size=(200, 200),                          # 모든 데이터 사이즈 통일
            batch_size=10,                                    # 훈련 시 10개씩 잘라서 훈련(fit에서 굳이 안 잘라도 됨)
            class_mode='binary',
            color_mode='grayscale',
            shuffle=True

    )   # Found 120 images belonging to 2 classes.

print(xy_train)             # <keras.preprocessing.image.DirectoryIterator object at 0x0000023F950CCD90>
print(xy_test)              # <keras.preprocessing.image.DirectoryIterator object at 0x000001E0A84DA0A0>

# 먼소리여..... 
print(xy_train[0][0].shape)         # (10, 200, 200, 1)                 => 배치사이즈 10으로 나눴으니까 16개 행에 (10, 200, 200, 1)씩 들어가 있음. 
print(xy_train[0][1].shape)         # (10,)
'''
x data: xy_train[0][0] ~ [15][0]까지
y data: xy_train[0][1] ~ [15][1]까지
'''

print(type(xy_train))               # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))            # <class 'tuple'>          리스트와의 차이점: 생성 후 데이터 변경(수정) 불가
print(type(xy_train[0][0]))         # <class 'numpy.ndarray'>
print(type(xy_train[0][1]))         # <class 'numpy.ndarray'>