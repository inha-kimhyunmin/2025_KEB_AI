# v0.9) v0.8파일의 결측치 값을 산술평균으로 채워 넣는 다양한 방법을 적용하시오.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

titanic = sns.load_dataset('titanic')
median_age = titanic['age'].median()
titanic_fill_row = titanic.fillna({'age':median_age}) #결측치 처리

X = titanic_fill_row[['age']] #x축 은 나이
Y = titanic_fill_row[['survived']] #y축은 생존 bool 타입
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #전체 데이터의 20%만 사용, 랜덤 시드 42

model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

plt.scatter(X_test, Y_test, color = 'red', label = 'age : survived test set')
plt.scatter(X_test, Y_pred, color = 'blue', label = 'Prediction for age:survived')
plt.xlabel('age')
plt.ylabel('survived')
plt.legend()
plt.show()
#