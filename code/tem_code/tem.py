'''
CoT_faithfulness_category.code.tem_code.tem 的 Docstring
从data/sort_by_domain/faithfulness_papers_general.csv种提取所有category不为benchmark或survey或other的论文
'''

import pandas as pd
import os

df1 = pd.read_csv('data/sort_by_category/faithfulness_papers_other.csv')
print(df1["Title"].tolist())

df2 = pd.read_csv('data/sort_by_domain/faithfulness_papers_general.csv')

# 过滤掉category为benchmark、survey或other的论文
filtered_df = df2[~df2['Category'].isin(['Benchmark', 'Survey', 'Other'])]

# 保存结果到新的CSV文件
filtered_df.to_csv('data/sort_by_domain/faithfulness_papers_filtered.csv', index=False)

