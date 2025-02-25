# v0.9) v0.8파일의 결측치 값을 산술평균으로 채워 넣는 다양한 방법을 적용하시오.

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer

df = pd.DataFrame(
    {
        'A':[1, 2, np.nan, 4], #1열
        'B':[np.nan, 12, 3, 4], #2열
        'C':[1, 2, 3, 4] #3열
    }
)


i = SimpleImputer(strategy='mean')
df[['A', 'B']] = i.fit_transform(df[['A', 'B']])

print(df)