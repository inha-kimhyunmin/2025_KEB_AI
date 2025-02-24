from KNeighborsRegressor import KNeighborsRegressor
import pandas as pd
import numpy as np

ls = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")

X = ls[["GDP per capita (USD)"]].values
y = ls[["Life satisfaction"]].values
print(ls)

model = KNeighborsRegressor(n_neighbors=5)

X_new = [[31721.3]]  # ROK 2020
model.fit(X_new, X, y)
print(model.predict())