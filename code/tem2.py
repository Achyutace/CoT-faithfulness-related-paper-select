import os
import pandas as pd

# 三个CSV文件路径
CSV_PATHS = [
    r"C:\Users\nattin\Desktop\论文\CoT_faithfulness_category\data\screened_papers_result_v1.csv",
    r"C:\Users\nattin\Desktop\论文\CoT_faithfulness_category\data\screened_papers_result_v2.csv",
    r"C:\Users\nattin\Desktop\论文\CoT_faithfulness_category\data\screened_papers_result_v3.csv"
]

def main():
    # 读取三个CSV文件
    dfs = []
    for csv_path in CSV_PATHS:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"✓ 读取 {os.path.basename(csv_path)}: {len(df)} 行")
            dfs.append(df)
        else:
            print(f"⚠️  文件不存在: {csv_path}")
    
    if not dfs:
        print("错误：没有找到任何CSV文件")
        return
    
    # 合并数据框并取并集（去重）
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"\n合并后共 {len(df_combined)} 行（去重前）")
    
    # 统计每个title出现的次数
    title_counts = df_combined['title'].value_counts().to_dict()
    
    # 根据title去重（保留第一次出现的）
    df_unique = df_combined.drop_duplicates(subset=['title'], keep='first')
    print(f"去重后共 {len(df_unique)} 行\n")
    
    # 打印所有 decision=REJECT 的行
    if 'decision' in df_unique.columns:
        reject_rows = df_unique[df_unique['decision'].astype(str).str.upper() == 'REJECT']
        
        if len(reject_rows) > 0:
            print(f"📋 共有 {len(reject_rows)} 行 decision=REJECT:")
            print("=" * 100)
            
            for idx, row in reject_rows.iterrows():
                title = row['title']
                count = title_counts.get(title, 0)
                print(f"\n[{idx}] Title: {title}")
                print(f"    出现次数: {count}")
                print(f"    is_cot: {row.get('is_cot', 'N/A')}")
                print(f"    is_post_hoc_only: {row.get('is_post_hoc_only', 'N/A')}")
                print(f"    reason: {row.get('reason', 'N/A')}")
                print("-" * 100)
        else:
            print("✓ 没有 decision=REJECT 的行")
    else:
        print("⚠️  警告：CSV文件中没有 'decision' 列")

if __name__ == "__main__":
    main()

