import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator                     # 이미지를 데이터로 변경 => 증폭 가능
from tensorflow.keras.datasets import fashion_mnist


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
augument_size = 40000                 # 40000장으로 증폭 위한 변수 생성
randidx = np.random.randint(x_train.shape[0], size=augument_size)                # x_train.shape[0] = 60000(fashion 총 데이터 개수 )
                                                                                 # 중에 랜덤하게 40000개를 추출한다.
# print(randidx)              # 랜덤한 40000개 데이터: [25945 16311 59551 ... 57673 37067 37648]
# print(len(randidx))         # 40000

x_augument = x_train[randidx].copy()                # x_train의 [randidx]번째 데이터를 x_agument에 저장!!, 혹시 모르니까 copy()로 데이터 보호
y_augument = y_train[randidx].copy()

# print(x_agument.shape, y_agument.shape)            # (40000, 28, 28) (40000,)

x_agument = x_augument.reshape(40000, 28, 28, 1)

train_datagen = ImageDataGenerator(
    rescale=1./255,             # 255로 나눈다 => min_max scaling
    horizontal_flip=True,       # 수평선을 기준으로 image 반전
    vertical_flip=True,         # 수직선을 기준으로 image 반전
    width_shift_range=0.1,      # 0.1만큼 이동
    height_shift_range=0.1,     # 0.1만큼 이동
    rotation_range=5,           # image 회전
    zoom_range=1.2,             # image 120% 확대
    shear_range=0.7,            # 임의 전단 변환 (shearing transformation) 범위
    fill_mode='nearest'         # 이미지 shift 시켰을 때 빈 공간 생기면 가까이에 있는 데이터로 채우기
)


#### 데이터 변환 ####
x_augumented = train_datagen.flow(                                           # flow_from_directory와 차이점 => 미리 수치화된 데이터를 가져와서 쓴다
       x_augument,
       y_augument,
       batch_size=augument_size,
       shuffle=True
    ) 

# print(x_data[0])
print(x_augumented[0][0].shape)               # (40000, 28, 28, 1)   => x
print(x_augumented[0][1].shape)               # (40000,)             => y

#### x_agumneted와 x_train 합치기 위한 x_train reshape ####
x_train = x_train.reshape(60000, 28, 28, 1)

### 데이터 합치기 ###
x_train = np.concatenate((x_train, x_augumented[0][0]))
y_train = np.concatenate((y_train, x_augumented[0][1]))


### 데이터 증가 완료 ###
print(x_train.shape, y_train.shape)                 # (100000, 28, 28, 1) (100000,)


