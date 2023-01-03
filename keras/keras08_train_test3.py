import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split            #사이킷런 트레인테스트스플릿 가져오긴


#1. data
x = np.array([1,2,3,4,5,6,7,8,9,10])        #(10, )
y = np.array(range(10))                     #(10, )       => weight = 1, bias = -1

#numpy 리스트 슬라이싱
# x_train = x[:7]             #index랑 data 값 헷갈리지 않기 / index= 0 ~ n-1
# x_test = x[7:]

# y_train = y[:7]             #y_train = y[:-2]
# y_test = y[7:]              #y_test = y[-3:]

#[search] train과 test를 섞어 7:3으로 만들기 (무작위 분할 추출)

x_train, x_test, y_train, y_test = train_test_split(
    
    #train set's parameter
    x, y, 
    # train_size=0.7, 
    test_size=0.3, 
    #shuffle=True,               # shuffle='False'면 순서대로 뽑힘 (단순히 list slicing) 
    #stratify = None, 
    random_state=123             #random number 표에 따라 동일한 랜덤 데이터 추출
    )

'''
test_size: 테스트 셋 구성의 비율을 나타냅니다. 
train_size의 옵션과 반대 관계에 있는 옵션 값이며, 주로 test_size를 지정해 줍니다. 
0.3는 전체 데이터 셋의 30%를 test (validation) 셋으로 지정하겠다는 의미입니다. 
default 값은 0.25 입니다.
shuffle: default=True 입니다. split을 해주기 이전에 섞을건지 여부입니다. 
보통은 default 값으로 놔둡니다.
stratify: default=None 입니다. classification을 다룰 때 매우 중요한 옵션값입니다. 
stratify 값을 target으로 지정해주면 각각의 class 비율(ratio)을 train / validation에 유지해 줍니다. 
(한 쪽에 쏠려서 분배되는 것을 방지합니다) 
만약 이 옵션을 지정해 주지 않고 classification 문제를 다룬다면, 성능의 차이가 많이 날 수 있습니다.
random_state: 세트를 섞을 때 해당 int 값을 보고 섞으며, 
하이퍼 파라미터를 튜닝시 이 값을 고정해두고 튜닝해야 
매번 데이터셋이 변경되는 것을 방지할 수 있습니다.  => 뒤에 지정한 숫자에 따라 뽑는 데이터셋 달라짐!


print('x train: ', x_train)
print('x test: ', x_test)
print('y train: ', y_train)
print('y test: ', y_test)
'''

#2. modeling
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(1))


#3. compile training
model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train, batch_size=1, epochs=100)         #훈련용 데이터로 fit


#4. evalu predict
loss = model.evaluate(x_test,y_test)            #평가용 데이터로  evaluate
print('loss: ', loss)
result = model.predict([11])
print('result: ', result)

