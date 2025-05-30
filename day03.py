import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#데이터 로딩 -> 데이터 전처리 -> 타겟 및 독립변수 설정 -> 트레이닝/테스트 셋 설정 -> 모델 설정 및 학습
# -> 예측 수행 -> 성능 평가 -> 시각화

#원하는 데이터셋을 불러오고
#원하는 x값과 y값(상관관계가 있을거 같은 데이터를 선택)
#모델을 가져와서 훈련 세트와 테스트 세트를 나눠서 훈련을 시키고
#테스트 세트에 적용시켜보기
#오차율 확인 -> 모델 검증

mpg = sns.load_dataset('mpg')
print(mpg)
print(mpg.info())

#mpg는 타겟(예측하는 데이터) 실제 측정결과는 정답지가 되겠죠
#일단 object 데이터는 제거, 그리고 결측치 제거
#decision tree, forest 모델 배우긴 했으나 모른다. 아는거는 선형회귀와 KNN

num_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']

# 그래프 설정
plt.figure(figsize=(12, 8))

# 각 변수와 mpg의 관계를 산점도로 표현 ,num_cols의 i번째 인덱스가 x, mpg가 y
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=mpg[col], y=mpg['mpg'])
    plt.xlabel(col)
    plt.ylabel('mpg')

plt.tight_layout()
plt.show()

x = mpg[['displacement']]
y = mpg[['mpg']]

#훈련 세트, 테스트 세트 분리
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

#직선 공간 생성, X_test의 가장 작은 값부터 가장 큰값까지, 100개의 구간으로 나눔 -> reshape를 통해 2차원 배열로 변환
X_range = np.linspace(X_test["displacement"].min(), X_test["displacement"].max(), 100).reshape(-1,1)
y_pred = model.predict(X_range)

plt.plot(X_range, y_pred, color = 'red', label = 'displacement : mpg model test')
plt.scatter(X_test, Y_test, color = 'blue', label = 'displacement : mpg ')
plt.xlabel('displacement')
plt.ylabel('mpg')
plt.legend()
plt.show()