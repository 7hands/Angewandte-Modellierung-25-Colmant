import pandas as pd

df2 = pd.read_csv('data_chars.csv')


df2.sort_values(by=['character'],
                    axis=0,
                    ascending=[False], 
                    inplace=True)

df2.to_csv('data_chars_sort.csv', index=False)