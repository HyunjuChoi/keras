import numpy as np
#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
'''input dimension = 2, output dimension = 1'''

x = np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])           #데이터셋 두개
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

print(x.shape)      #행렬(데이터)의 구조: (2,10) = > 2X10 행렬 상태  
                    # => 10X2로 만들려면 [[1,1], [2,1] ... ] 형태로 만들어야 됨
print(y.shape)      #(10,)

x = x.T             #행,열 바꿈
print(x.shape)      # (10,2)
