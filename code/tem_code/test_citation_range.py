'''
code.tem_code.test_citation_range 的 Docstring
看下有多少篇论文引用了faithfulness那篇，但没引用CoT faithfulness两篇，而且算faithfulness的文章
我服了这个草台班子。。。。。。
'''


import pandas as pd
import requests
import time
import re

# ================= 配置区域 =================
INPUT_FILE = 'data/papers_with_dimensions_v1.csv'
OUTPUT_FILE = 'data/papers_filtered_smart.csv'

PAPER_TARGET = "Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?"

PAPER_BAD_LIST = [
    "Measuring Faithfulness in Chain-of-Thought Reasoning",
    # 加上副标题可以防止搜到重名的博客或评论文章
    "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting"
]

API_URL_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
API_URL_CITATIONS = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
# ===========================================

def normalize_title(title):
    """
    标准化标题：转小写，去除标点和空格，用于跨来源比对
    例如: "The  Cat." -> "thecat"
    """
    if not isinstance(title, str):
        return ""
    return re.sub(r'[^a-z0-9]', '', title.lower())

def get_paper_id(title):
    """根据标题搜索获取 S2ID"""
    params = {'query': title, 'limit': 1, 'fields': 'paperId,title'}
    try:
        resp = requests.get(API_URL_SEARCH, params=params, timeout=10)
        data = resp.json()
        if data['total'] > 0:
            return data['data'][0]['paperId'], data['data'][0]['title']
    except Exception as e:
        print(f"搜索出错 {title}: {e}")
    return None, None

def get_citing_papers(paper_id, paper_name):
    """
    获取引用了 paper_id 的所有论文 ID。
    如果是白名单论文，我们需要返回详细信息（ID和Title）；
    如果是黑名单论文，我们只需要 ID 集合用来做减法。
    """
    citing_ids = set()
    citing_details = {} # 仅用于白名单: {normalized_title: original_title}
    
    offset = 0
    limit = 1000 # 单次最大请求
    
    print(f"正在获取引用了 <{paper_name}> 的论文列表...", end="")
    
    while True:
        # 获取引用这篇论文的论文 (citingPapers)
        params = {
            'fields': 'paperId,title',
            'offset': offset,
            'limit': limit
        }
        try:
            resp = requests.get(API_URL_CITATIONS.format(paper_id=paper_id), params=params)
            if resp.status_code != 200:
                print(f" API Error: {resp.status_code}")
                break
                
            data = resp.json()
            papers = data.get('data', [])
            
            if not papers:
                break
                
            for p in papers:
                citing_paper = p.get('citingPaper')
                if citing_paper and citing_paper.get('paperId'):
                    pid = citing_paper['paperId']
                    citing_ids.add(pid)
                    
                    # 同时记录标题，用于最后和本地CSV匹配
                    ptitle = citing_paper.get('title')
                    if ptitle:
                        citing_details[pid] = ptitle
            
            print(f".", end="") # 进度点
            
            if len(papers) < limit:
                break # 没有更多数据了
            offset += limit
            time.sleep(1) # 避免速率限制
            
        except Exception as e:
            print(f" Error: {e}")
            break
            
    print(f" 完成 (共 {len(citing_ids)} 篇)")
    return citing_ids, citing_details

# ================= 主逻辑 =================

# 1. 获取核心论文的 ID
target_id, target_real_name = get_paper_id(PAPER_TARGET)
if not target_id:
    raise Exception(f"找不到目标论文: {PAPER_TARGET}")

bad_ids = []
for bad_name in PAPER_BAD_LIST:
    bid, _ = get_paper_id(bad_name)
    if bid:
        bad_ids.append((bid, bad_name))
    else:
        print(f"警告: 找不到黑名单论文: {bad_name}")

print("-" * 30)

# 2. 构建集合
# A: 引用了 Target 的论文集合
set_white_ids, dict_white_titles = get_citing_papers(target_id, target_real_name)

# B & C: 引用了 Bad 论文的论文集合
set_black_ids = set()
for bid, bname in bad_ids:
    c_ids, _ = get_citing_papers(bid, bname)
    set_black_ids.update(c_ids)

# 3. 集合减法 (在白名单 但 不在黑名单)
valid_ids = set_white_ids - set_black_ids
print(f"\n逻辑计算完成:")
print(f"- 引用了 Target 的论文数: {len(set_white_ids)}")
print(f"- 引用了 Bad List 的论文数(并集): {len(set_black_ids)}")
print(f"- 最终符合 citation 逻辑的论文数 (全网范围): {len(valid_ids)}")

# 4. 准备本地匹配字典
# 我们需要将 API 返回的有效论文标题做成一个查找表 (Normalized Title -> True)
valid_titles_normalized = set()
for pid in valid_ids:
    if pid in dict_white_titles:
        valid_titles_normalized.add(normalize_title(dict_white_titles[pid]))

# 5. 读取本地 CSV 并筛选
print("-" * 30)
print("正在与本地 CSV 进行匹配...")
df = pd.read_csv(INPUT_FILE)
original_count = len(df)

# 定义筛选函数
def is_in_valid_list(row_title):
    return normalize_title(row_title) in valid_titles_normalized

# 应用筛选
matched_df = df[df['title'].apply(is_in_valid_list)]

print(f"本地 CSV 总数: {original_count}")
print(f"筛选后剩余数: {len(matched_df)}")

if len(matched_df) > 0:
    print("\n匹配到的前 5 篇:")
    for t in matched_df['title'].head(5):
        print(f"- {t}")
    matched_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n结果已保存至: {OUTPUT_FILE}")
else:
    print("\n未在本地 CSV 中找到符合引用逻辑的论文。")