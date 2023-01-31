import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator                     # 이미지를 데이터로 변경 => 증폭 가능
from tensorflow.keras.datasets import fashion_mnist


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
augument_size = 100                 # 100장으로 증폭 위한 변수 생성


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



x_data = train_datagen.flow(                                           # flow_from_directory와 차이점 => 미리 수치화된 데이터를 가져와서 쓴다
       # 이미지 데이터를 지정한 타일 개수만큼 붙임    
       np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1),    # x data   (-1 = 전체데이터 (현재는 100개를 의미))
       np.zeros(augument_size),                                                     # y data, 0: 100개
       batch_size=augument_size,
       shuffle=True
    ) 

# print(x_data[0])
print(x_data[0][0].shape)               # (100, 28, 28, 1)
print(x_data[0][1].shape)               # (100,)


import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][1], cmap='gray')
plt.show()