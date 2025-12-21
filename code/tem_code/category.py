'''
CoT_faithfulness_category.code.category 的 Docstring
用来处理一开始打标的那些csv

'''
import os
import csv
import pandas as pd

# 设置目录
DATA_PATH = os.path.dirname(os.path.dirname(__file__)) + '/data/'
print(DATA_PATH)

df = pd.read_csv(DATA_PATH + 'faithfulness_papers_full_survey.csv')

def sort_by_X(dimension, kinds):
    '''
    sort_by_X 的 Docstring
    根据指定维度对论文分类
    param dimension: 分类维度，如 'Category' 或 'Field'
    param kinds: 该维度的所有可能取值列表
    return: None，直接在全局 df 上操作并保存分类结果
    '''
    for kind in kinds:
        df_kind = df[df[dimension] == kind]
        output_file = DATA_PATH + f'sort_by_{dimension.lower()}/faithfulness_papers_{kind.lower()}.csv'
        df_kind.to_csv(output_file, index=False)
        print(f"Saved {len(df_kind)} papers to {output_file}")
    
sort_by_X('Domain', ['Math', 'Code', 'Medical', 'General', 'Logic'])
sort_by_X('Category', ['Phenomenon', 'Survey', 'Metric', 'Benchmark', 'Mitigation', 'Other'])


