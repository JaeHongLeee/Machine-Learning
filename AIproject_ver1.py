import matplotlib.pyplot as plt
import numpy as np
import sklearn

from sklearn import datasets
from sklearn.datasets import load_diabetes

from keras import optimizers
from keras.layers import Dense, Activation
from keras.models import Sequential

diabetse = load_diabetes()
print(diabetse.keys())
print(diabetse.feature_names)
aa=diabetse.data
bb=diabetse.target

nr, nc = aa.shape

nov=400
x_train = aa[:nov, :]
y_train = bb[:nov]

model = Sequential()
model.add(Dense(8, input_dim=nc,activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.summary()

H=model.fit(x_train, y_train, batch_size= 20, epochs=500)

plt.figure()
plt.title('loss')
plt.plot(H.history['loss'])

y_predict=model.predict(aa[:,:])
plt.figure()
plt.title('disease progression one year after baseline')
plt.plot(bb[:],'g',label='real')
plt.plot(y_predict,'r',label='predict')
plt.legend()
plt.show()
