import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. data
x = np.array([range(10), range(21, 31), range(201, 211)]) 
y = np.array([[1,2,3,4,5,6,7,8,9,10],
                [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4] ])               # => x 특성 개수와  y 특성 개수 달라도 가능

#print(x.shape)          #(3, 10)

x = x.T
#print(x.shape)
y = y.T

#[실습] train_test_split을 이용하여 
#7:3 비율로 잘라서 모듈 구현 / 소스 완성

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.3, 
                                                    shuffle=True, 
                                                    #stratify=y, 
                                                    random_state=123)
                                   

print('x train: ', x_train)
print('x test: ', x_test)
print('y train: ', y_train)
print('y test: ', y_test)

