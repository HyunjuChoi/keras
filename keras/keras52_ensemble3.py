import numpy as np
from sklearn.metrics import r2_score, accuracy_score  # Accuracy Score 추가

# 데이터 준비
x1_datasets = np.array([range(100), range(301,401)]).transpose()            # 삼성전자 시가, 고가 데이터
# print(x1_datasets)      # (100, 2) 

x2_datasets = np.array([range(101,201), range(411, 511), range(150, 250)]).T        # 아모레 시가, 고가, 종가 데이터
# print(x2_datasets)              # (100, 3)

x3_datasets = np.array([range(100,200), range(1301,1401)]).T
# print(x3_datasets.shape)        # (100, 2)

y1 = np.array(range(2001, 2101))           # (100, )       # 삼성전자의 하루 뒤 종가
y2 = np.array(range(201, 301))              # (100, )          # 아모레의 하루 뒤 종가


# 실습 
from sklearn.model_selection import train_test_split

(x1_train, x1_test, x2_train, x2_test, 
 x3_train, x3_test, 
 y1_train, y1_test, y2_train, y2_test) = train_test_split(x1_datasets, x2_datasets, x3_datasets, y1, y2, 
                                                          shuffle=True, random_state=115, train_size=0.7)

# print(x1_train.shape, x2_train.shape, x3_train.shape, y1_train.shape, y2_train.shape)            # (70, 2) (70, 3) (70, 2) (70,) (70,)
# print(x2_test.shape, x2_test.shape, x3_test.shape, y1_test.shape, y2_test.shape)               # (70, 2) (70, 3) (30, 2) (30,) (30,)                        

# 2. modeling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

# 2-1. model 1 (x1_datasets)
input1 = Input(shape=(2, ))
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
output1 = Dense(14, activation='relu', name='ds14')(dense3)

# 2-2. model 2 (x2_datasets)
input2 = Input(shape=(3, ))
dense21 = Dense(21, activation='linear', name='ds21')(input2)
dense22 = Dense(22, activation='linear', name='ds22')(dense21)
output2 = Dense(23, activation='linear', name='ds23')(dense22)

# 2-3. model 3 (x3_datasets)
input3 = Input(shape=(2, ))
dense31 = Dense(11, activation='relu', name='ds31')(input3)
dense32 = Dense(12, activation='relu', name='ds32')(dense31)
dense33 = Dense(13, activation='relu', name='ds33')(dense32)
output3 = Dense(14, activation='relu', name='ds34')(dense33)

# 2-4. merge model
from tensorflow.keras.layers import concatenate, Concatenate

merge1 = Concatenate()([output1, output2, output3])
# merge1 = concatenate([output1, output2, output3], name='mg1')
merge2 = Dense(12, activation='relu', name='mg2')(merge1)
merge3 = Dense(13, name='mg3')(merge2)
last_output = Dense(1, name='last1')(merge3)                         # y 데이터 컬럼이 1이므로 마지막 레이어 dense = 1

# 2-5-1. 모델 분기1
dense41 = Dense(12, activation='relu', name='ds41')(last_output)
dense42 = Dense(13, activation='relu', name='ds42')(dense41)
dense43 = Dense(14, activation='relu', name='ds43')(dense42)
output4 = Dense(1, name='last2')(dense43)

# 2-5-2. 모델 분기2
dense51 = Dense(12, activation='relu', name='ds51')(last_output)
dense52 = Dense(13, activation='relu', name='ds52')(dense51)
dense53 = Dense(14, activation='relu', name='dse53')(dense52)
output5 = Dense(1, name='last3')(dense53)


model = Model(inputs= [input1, input2, input3], outputs=[output4, output5])

# model.summary()

# 3. Compile and Training
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=300, batch_size=8)

# 4. evaluation and prediction
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])           # mse
print('loss: ', loss)

y_predict = model.predict([x1_test, x2_test, x3_test])

# print(y_test.shape)             # (30,)
# print(y_predict.shape)          # (30, 1)

# y_predict = y_predict.reshape(30, )

# acc = accuracy_score(y_test, y_predict)
# print('acc: ', acc)



'''
1/30 앙상블 모델 3개,  output 2개

1.
1/1 [==============================] - 0s 160ms/step - loss: 0.4531 - last1_loss: 0.4478 - last2_loss: 0.0053
loss:  [0.4530600905418396, 0.447774738073349, 0.0052853417582809925]


'''