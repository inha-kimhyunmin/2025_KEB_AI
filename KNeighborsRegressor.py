import numpy as np


# 주어진 입력 데이터의 주변 k개의 이웃 데이터의 평균을 내어 예측을 하는 방식
class KNeighborsRegressor:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.nearest_neighbors = None
        self.sorted_indices = None
        self.x = None
        self.y = None

    def fit(self, x_new, x, y):
        """
        새로 입력된 x를 기준으로 가장 가까운 k개의 x값들에 해당하는 y값들 구하는 함수
        :param x: independent variable list(x-axis)
        :param y: dependent variable list(y-axis)
        :return:
        """
        self.x = x
        self.y = y
        distance = np.abs(x - x_new)
        self.sorted_indices = np.argsort(distance, axis=0)
        self.nearest_neighbors = y[self.sorted_indices[:self.n_neighbors]]

    def predict(self):
        return np.mean(self.nearest_neighbors)

    def predict_by_repetition(self):
        sum = 0
        for i in range(self.n_neighbors):
            sum += self.y[self.sorted_indices[i][0]][0]
        print(sum / 5)
