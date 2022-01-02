import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def func(x, a, b):
    return a*np.exp(b*x)    # a*e^(b*x)

day = np.arange(1, 34)  #1~32까지 배열
cases = np.array([5123, 5266,4944, 5352, 5128, 4325, 4954, 7175, 7102, 7022, 6977, 6689, 5817, 5567, 7850, 7622,
                  7435, 7314,6236, 5318, 5202, 7456, 6919, 6233, 5842, 5419, 4207, 3865, 5409,5037,4875,4416, 3830])
                #12월 1일 ~ 1월 2일 까지 확진자 수
                #리스트를 배열로 바꾸어서 메모리를 적게 차지하고 연산이 빠르도록 만듬
plt.plot(day, cases, 'co', label='Total Number of Cases in korea')  #데이터를 점으로 찍어냄

popt, pcov = curve_fit(func, day, cases)    #popt는 피팅결과 즉 피팅으로 알아낸 파라미터 a,b를 결과로 변환
                                            #pcov는 얼마나 잘 피팅되었는지 판단하는 변수
plt.plot(day, func(day, *popt), 'r-', label='fit: a = %5.3f, b = %5.3f' %tuple(popt))    #데이터에 가장 가까운 선형 그래프를 그림

plt.xlabel('Prediction of confirmed cases on January 3')
plt.ylabel('Total Number of Cases in korea')
plt.legend()
plt.show()
