'''

'filter_papers 的 Docstring
12月25号写的，没想到还挺好用，直接过滤掉那些死掉的论文，留下那些活着的论文
根据 alive_filtered_papers.csv 中的标题，筛选出 filtered_faithfulness_papers_v1.csv 中的论文
'''
import os
import pandas as pd

# 配置路径
META_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = META_PATH + "/data/"

ALIVE_CSV = DATA_PATH + "alive_filtered_papers.csv"
SOURCE_CSV = DATA_PATH + "filtered_faithfulness_papers_v1.csv"
OUTPUT_CSV = DATA_PATH + "filtered_faithfulness_papers_v2.csv"

def main():
    # 1. 读取 alive_filtered_papers.csv 的 Title 列
    print(f"读取 {ALIVE_CSV}...")
    alive_df = pd.read_csv(ALIVE_CSV)
    
    # 获取 Title 列（注意大小写）
    if 'Title' in alive_df.columns:
        target_titles = set(alive_df['Title'].dropna().str.strip())
    elif 'title' in alive_df.columns:
        target_titles = set(alive_df['title'].dropna().str.strip())
    else:
        raise ValueError("alive_filtered_papers.csv 中找不到 'Title' 或 'title' 列")
    
    print(f"找到 {len(target_titles)} 个目标标题")
    
    # 2. 读取 filtered_faithfulness_papers_v1.csv
    print(f"读取 {SOURCE_CSV}...")
    source_df = pd.read_csv(SOURCE_CSV)
    
    print(f"源文件共有 {len(source_df)} 行")
    
    # 3. 匹配对应的行（不区分大小写，去除首尾空格）
    source_df['title_normalized'] = source_df['title'].astype(str).str.strip()
    target_titles_normalized = {title.strip() for title in target_titles}
    
    # 创建小写映射用于大小写不敏感匹配
    target_titles_lower_map = {title.lower(): title for title in target_titles_normalized}
    
    # 匹配（精确匹配和忽略大小写匹配）
    matched_rows = []
    matched_titles = set()
    
    for idx, row in source_df.iterrows():
        title_normalized = row['title_normalized']
        # 精确匹配
        if title_normalized in target_titles_normalized:
            matched_rows.append(row)
            matched_titles.add(title_normalized)
        else:
            # 尝试忽略大小写匹配
            title_lower = title_normalized.lower()
            if title_lower in target_titles_lower_map:
                matched_rows.append(row)
                matched_titles.add(title_normalized)
    
    print(f"匹配到 {len(matched_rows)} 行")
    
    if len(matched_rows) == 0:
        print("警告：没有匹配到任何行")
        return
    
    # 转换为 DataFrame
    result_df = pd.DataFrame(matched_rows)
    
    # 删除临时的 normalized 列
    if 'title_normalized' in result_df.columns:
        result_df = result_df.drop(columns=['title_normalized'])
    
    # 4. 按 title 的首字母排序
    print("按 title 首字母排序...")
    # 创建一个辅助列用于排序（提取首字母，处理可能的空值和数字）
    result_df['_sort_key'] = result_df['title'].astype(str).str.strip().str[0].str.upper()
    result_df = result_df.sort_values(by='_sort_key', na_position='last')
    # 删除辅助列
    result_df = result_df.drop(columns=['_sort_key'])
    
    # 5. 保存到 filtered_faithfulness_papers_v2.csv
    result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"已保存 {len(result_df)} 行到 {OUTPUT_CSV}")
    
    # 显示未匹配的标题（用于检查）
    # 找出哪些目标标题没有被匹配到（通过小写比较）
    matched_titles_lower = {title.lower() for title in matched_titles}
    unmatched = [title for title in target_titles_normalized 
                 if title.lower() not in matched_titles_lower]
    
    if len(unmatched) > 0:
        print(f"\n警告：有 {len(unmatched)} 个标题未匹配到：")
        for title in sorted(list(unmatched))[:10]:  # 只显示前10个
            print(f"  - {title}")
        if len(unmatched) > 10:
            print(f"  ... 还有 {len(unmatched) - 10} 个未显示")

if __name__ == "__main__":
    main()

