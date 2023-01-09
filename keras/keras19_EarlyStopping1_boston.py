from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#그래프 한글 깨짐 방지
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'                   #폰트 저장된 경로에서 불러오기
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)


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
model.add(Dense(5, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. compile and training
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping                    #EarlyStopping 추가

#earlystopping 기준 설정
earlyStopping = EarlyStopping(
    monitor='val_loss',             #history의 val_loss의 최소값을 이용함
    mode='min',                     #max로 설정 시 갱신 안됨. (accuracy 사용 시에는 정확도 높을수록 좋기 때문에 max로 설정)
    patience=15,                     #earlystopping n번 (최저점 나올 때까지 n번 돌림. 그 안에 안 나오면 종료) 
    restore_best_weights=True,          #이걸 설정해줘야 종료 시점이 아닌 early stopping 지점의 최적 weight 값 사용 가능
    verbose=1
)

hist = model.fit(x_train, y_train, epochs=200, batch_size=1, 
          validation_split=0.2, verbose=1, callbacks=[earlyStopping])           #val_loss를 기준으로 최소값이 n번 이상 갱신 안되면 훈련 중지                 

#4. evalutaion and prediction
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('===================================')
#print(hist)                         # <keras.callbacks.History object at 0x0000024DDE601AC0>
print('===================================')
#print(hist.history)                 #loss 변화량 값 dictionary형으로 출력 (key:value) 키-값 쌍형태, value-리스트형
'''
{'loss': [7948.48779296875, 963.1211547851562, 383.9888610839844, 
246.97784423828125, 184.3928680419922, 152.10592651367188, 125.17009735107422, 
111.9476089477539, 100.05902099609375, 90.64845275878906], 

'val_loss': [1362.2457275390625, 390.7475280761719, 211.93231201171875, 
153.6135711669922, 123.99478149414062, 102.72587585449219, 91.13431549072266, 
83.18538665771484, 83.1338882446289, 69.14740753173828]}
'''
#print(hist.history['loss'])                 #loss값만 출력


#5. 시각화
plt.figure(figsize=(9,6))           #그래프 사이즈 설정
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')                #maker=선 무늬
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()                          #격자무늬 추가
plt.xlabel('에포크')                 #축 이름 추가
plt.ylabel('손실')
plt.title('보스턴 손실 그래프')            #그래프 이름 추가
# plt.legend()                      #선 이름(label) 출력, 그래프 없는 지점에 자동 설정
plt.legend(loc='upper right')       #loc=' ': 원하는 위치에 설정 가능
plt.show()




'''결과치 
1.Epoch 00059: early stopping
4/4 [==============================] - 0s 0s/step - loss: 27.3670
loss:  27.36704444885254
===================================
===================================


2.patience = 15로 수정
Epoch 00152: early stopping
4/4 [==============================] - 0s 0s/step - loss: 99.3449
loss:  99.34488677978516
===================================
===================================

'''