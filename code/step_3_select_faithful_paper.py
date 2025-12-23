'''
code.tem 的 Docstring
筛选所有标题和摘要不包含faithfulness、Unfaithfulness、faithful、Unfaithful的论文
'''

import os
import pandas as pd
META_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(META_PATH, 'data')

def get_csv_columns(file_path, column_name='title'):
    # 获取指定列的所有值
    df = pd.read_csv(file_path, usecols=column_name)
    return df[column_name].tolist()

def filter_papers(file_path):
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return
    
    # 1. 加载数据
    df = pd.read_csv(file_path)
    
    # 2. 定义关键词
    keywords = ['faithfulness', 'unfaithfulness', 'faithful', 'unfaithful']
    
    # 3. 创建筛选逻辑 (矢量化操作)
    # 将标题和摘要合并，全部转为小写
    content = (df['title'].fillna('') + ' ' + df['abstract'].fillna('')).str.lower()
    
    # 检查是否包含任何一个关键词
    # join 结果如: "faithfulness|unfaithfulness|..."
    pattern = '|'.join(keywords)
    
    # 筛选出【包含】这些关键词的行
    filtered_df = df[content.str.contains(pattern, case=False, na=False)]
    
    print(f"原始数据量: {len(df)}")
    print(f"筛选后数据量: {len(filtered_df)}")
    
    return filtered_df
def main():
    # file_path = os.path.join(DATA_PATH, 'sort_by_domain/sort_by_has_phenomenon/faithfulness_papers_true_v1.csv')
    file_path = os.path.join(DATA_PATH, 'papers_with_dimensions_v1.csv')
    # file_path = os.path.join(DATA_PATH, 'papers_with_dimensions_v1.csv')
    filtered_df = filter_papers(file_path)
    filtered_df.to_csv(os.path.join(DATA_PATH, 'filtered_fairness_papers.csv'), index=False)
if __name__ == "__main__":
    main()