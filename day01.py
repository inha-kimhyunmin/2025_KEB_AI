import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from tglearn import LinearRegression

ls = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")

X = ls[["GDP per capita (USD)"]].values
y = ls[["Life satisfaction"]].values
print(ls)
ls.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23500, 62500, 4, 9])

#선형 회귀 모델 정의
model =  LinearRegression()
model.fit(X,y) #x절편, y절편 구함

X_range = np.linspace(23500, 62500, 100).reshape(-1, 1)  # X 값 범위
y_pred = model.predict(X_range)

# 회귀 직선 그리기
plt.plot(X_range, y_pred, color='red', label="Regression Line")

# 그래프 제목과 레이블 추가
plt.title("Life Satisfaction vs GDP per Capita")
plt.xlabel("GDP per capita (USD)")
plt.ylabel("Life satisfaction")

x_new = [[31721.3]]
y_new_pred = model.predict(x_new)

plt.scatter(x_new, y_new_pred, color = 'green', label = f"Prediction for X={x_new[0][0]} : Y={y_new_pred[0]}")
# 범례 추가
plt.legend()

# 그래프 표시
plt.show()