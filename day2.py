import pandas as pd

data = [1, 7, 5, 2, 8, 3, 6, 4]

bins = [0, 3, 6, 9]

labels = ["low", "mid", "high"]

#0 ~ 3 low 4 ~ 6 mid 7 ~ 9 high
#매개변수를 순서대로 집어넣던가, 키워드 매개변수로 원하는대로 집어넣는것도 가능
cat = pd.cut(data, bins, True, labels) #False하면 3이 mid, 6이  high가 된다.
print(cat)
