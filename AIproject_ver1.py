# 사용 라이브러리 import
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from sklearn import datasets
from sklearn.datasets import load_diabetes

from keras import optimizers
from keras.layers import Dense, Activation
from keras.models import Sequential

# 442 명의 당뇨병 환자 가각에 대해 10개의 기준 변수, 연령, 성별, 체질량 지수, 평균 혈압 및 6개의 혈청 측정 값
diabetse = load_diabetes()
print(diabetse.keys())  # key()함수는 해당 사전형 데이터에 저장된 key 값들을 리스트의 형태로 변환
print(diabetse.feature_names)   # featur_names라는 key값에는 age, sex, bmi, bp, s 등등으 10가지 기준 변수가 있다.
aa=diabetse.data    # 442x10 행렬의 데이터 (행은 환자번호, 열은 기준 변수)
bb=diabetse.target  # 1년 후 질병 진행에 대한 실제 정량적 측정

nr, nc = aa.shape   #442, 10

nov=400 # 학습할 데이터
x_train = aa[:nov, :]   # 입력 데이터
y_train = bb[:nov]  #라벨 값

model = Sequential()    # Seqeuntial 모델 오브젝트를 model이라는 변수안에 넣고 모델 구성
model.add(Dense(8, input_dim=nc,activation='relu')) # 출력 뉴런(노드), 입력 뉴런(노드), 활성화 함수
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1)) # 마지막 레이어의 노드갯수는 출력 데이터 갯수와 동일하게 1로 지정
# relu는 은닉층, Dense 레이어는 입력과 출력을 모두 연결해주며 입력과 출력을 각각 연결해주는 가중치를 포함한다.

model.compile(loss='mse', optimizer='adam') #모델 컴파일
# loss는 입력데이터가 출력데이터와 일치하는지 평가해주는 손실함수 (평균제곱오차mse(mean squared error)를 사용)
# optimizer는 손실함수를 기반으로 네트워크가 어떻게 업데이트 될지 결정 (adam 사용)
model.summary() #모델의 구조를 요약해 출력

H=model.fit(x_train, y_train, batch_size= 20, epochs=500)
# 컴파일 한 모델을 학습
# epochs는 학습 반복횟수
# batch_size는 몇개의 샘플로 가중치를 갱신할 것인지 지정
# x_train,y_train,batch_size=20,epochs=500 뜻은 쉽게말하면
# 문제집(nov개의 문항)을 500회(epochs) 반복해서 푸는데
# 20문항(batch_size)을 풀때마다 해답을 맞추는 것을 뜻한다.

# 그래프로 출력하는 부분
plt.figure()
plt.title('loss')
plt.plot(H.history['loss']) #손실함수 플롯

y_predict=model.predict(aa[:,:])
plt.figure()
plt.title('disease progression one year after baseline')
plt.plot(bb[:],'g',label='real')    # 실제 값
plt.plot(y_predict,'r',label='predict') #예측 값
plt.legend()
plt.show()
