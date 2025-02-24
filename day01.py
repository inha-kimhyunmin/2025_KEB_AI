import numpy as np
import pandas as pd

a = np.array([1,2,3,4])
print(a)

df = pd.DataFrame(
    [4,7,10],
    [5,8,11],
    [6,9,12], index = [1,2,3], columns=['a','b','c']
)

