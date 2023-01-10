from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path = 'C:/Windows/Fonts/malgun.ttf'   

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
#model.add(Dense(5, input_dim=13))                  #input_dim은 행과 열일 때만 표현 가능함
model.add(Dense(10, input_shape=(13, )))             #(13, ), input_shape: 다차원일 때 input_dim 대신 사용!
                                                    #if (100, 10, 5)라면 (10,5)로 표현됨. 맨 앞의 100은 데이터 개수
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

#3. compile and training
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, 
          validation_split=0.2, verbose=1)                 

#4. evalutaion and prediction
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('===================================')
print(hist)                         # <keras.callbacks.History object at 0x0000024DDE601AC0>
print('===================================')
print(hist.history)                 #loss 변화량 값 dictionary형으로 출력 (key:value) 키-값 쌍형태, value-리스트형
'''
{'loss': [7948.48779296875, 963.1211547851562, 383.9888610839844, 
246.97784423828125, 184.3928680419922, 152.10592651367188, 125.17009735107422, 
111.9476089477539, 100.05902099609375, 90.64845275878906], 

'val_loss': [1362.2457275390625, 390.7475280761719, 211.93231201171875, 
153.6135711669922, 123.99478149414062, 102.72587585449219, 91.13431549072266, 
83.18538665771484, 83.1338882446289, 69.14740753173828]}
'''
print(hist.history['loss'])                 #loss값만 출력


#5. 시각화
plt.figure(figsize=(9,6))           #그래프 사이즈 설정
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')                #marker=선 무늬
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()                          #격자무늬 추가
plt.xlabel('epoch')                 #축 이름 추가
plt.ylabel('loss')
plt.title('보스턴 loss')            #그래프 이름 추가
# plt.legend()                      #선 이름(label) 출력, 그래프 없는 지점에 자동 설정
plt.legend(loc='upper right')       #loc=' ': 원하는 위치에 설정 가능
plt.show()
