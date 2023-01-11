import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



#1. data
datasets = load_digits()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)                             # (1797, 64) (1797,)
print(np.unique(y, return_counts=True))             # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),               =>다중분류 data
                                                    #  array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[3])
# plt.show()

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

# print(y)
# print(y.shape)                  #(1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=115, 
                                                    test_size= 0.2, stratify = y)

# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.modeling
model = Sequential()
model.add(Dense(30, activation='linear', input_shape=(64, )))
model.add(Dense(20, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='linear'))
model.add(Dense(10, activation='softmax'))

#3. compile and training
from tensorflow.keras.callbacks import EarlyStopping

earlyStopping = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=50,
    #baseline='0.1',                특정 값 도달 시 훈련 중지                       
    restore_best_weights=True,
    verbose=1
)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=5,
          validation_split=0.2, verbose=1, callbacks=[earlyStopping])

#4. evaluation and prediction
loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy: ', accuracy)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)

y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test, y_predict)

print('acc: ', acc)


'''
1/11일

<<min max>>
Epoch 00064: early stopping
12/12 [==============================] - 0s 1ms/step - loss: 0.2540 - accuracy: 0.9444
loss:  0.25403422117233276
accuracy:  0.9444444179534912
acc:  0.9444444444444444

<<standard>>
Epoch 00060: early stopping
12/12 [==============================] - 0s 635us/step - loss: 0.2434 - accuracy: 0.9500
loss:  0.2433536946773529
accuracy:  0.949999988079071
acc:  0.95
'''