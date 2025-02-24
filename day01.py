import pandas as pd

# s = pd.Series([1,2,3,4])
# s = pd.Series([1,2,3,4], index=['a','b','c','d'])
# print(s)
# print()
#
# s2 = pd.Series([99,100,98,91,92])
# s2_subset = s2[1:4]
# print(s2_subset)
# print()
# s2_mean = s2.mean()
# print(s2_mean, s2.min())

df = pd.DataFrame([
    [4,7,10],
    [5,88,11],
    [16,9,12]], index = [1,2,3], columns =['a', 'b', 'c'])
print(df)
print()
df2 = pd.melt(df).rename(columns = {'variable':'var','value':'val'}).query('val>10').sort_values('val', ascending=False)
print(df2)
print()
df3 = df.iloc[1:2]
print(df3)
print()
df4 = df.loc[1:2]
print(df4)
print()
df5 = df.iloc[:,[0,2]]
print(df5)
