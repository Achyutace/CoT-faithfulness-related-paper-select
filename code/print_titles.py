'''
code.tem_code.tem1 的 Docstring
列出来某个csv的所有title，方便喂给ai
'''
import os
import pandas as pd
META_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(META_PATH, 'data')

def get_csv_titles(file_path):
    # 获取title列所有title
    df = pd.read_csv(file_path, usecols=['title'])
    return df['title'].tolist()

def main():
    file_path = os.path.join(DATA_PATH, 'sort_by_domain/sort_by_has_phenomenon/faithfulness_papers_true_v1.csv')
    titles = get_csv_titles(file_path)
    print(titles)

if __name__ == "__main__":
    main()