import os
import pandas as pd

DATA_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/data/"
def sort_by_X(dimension, kinds, version="v1", csv_prefix = "papers_with_dimensions_", dir_p = DATA_PATH):
    '''
    sort_by_X 的 Docstring
    根据指定维度对论文分类
    param dimension: 分类维度，如 'Category' 或 'Field'
    param kinds: 该维度的所有可能取值列表
    return: None，直接在全局 df 上操作并保存分类结果
    '''
    # 创建分类文件夹
    folder_name = f'sort_by_{dimension.lower()}/'
    dir_path = os.path.join(dir_p, folder_name)
    df = pd.read_csv(dir_p+ f"{csv_prefix}{version}.csv")
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for kind in kinds:
        # --- 修复核心逻辑 ---
        # 1. 确保 kind 的类型与列中数据类型一致
        # 2. 进行向量化筛选
        if isinstance(kind, str) and kind.lower() in ["true", "false"]:
            # 处理可能的字符串/布尔值混淆
            actual_kind = True if kind.lower() == "true" else False
            df_kind = df[df[dimension].astype(str).str.lower() == kind.lower()]
        else:
            # 修改逻辑为re字符串匹配，字符串中存在kind
            df_kind = df[df[dimension].astype(str).str.contains(str(kind), na=False)]

            

        # 保存文件
        file_name = f'faithfulness_papers_{str(kind).lower()}_{version}.csv'
        output_file = os.path.join(dir_path, file_name)
        
        df_kind.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"维度 [{dimension}] 下的值 [{kind}]: 已保存 {len(df_kind)} 篇论文至 {output_file}")

if __name__ == "__main__":
    sort_by_X("has_phenomenon", ["true", "false"], version="v1", csv_prefix = "faithfulness_papers_general_", dir_p=DATA_PATH+"/sort_by_domain/")