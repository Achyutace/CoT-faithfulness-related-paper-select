import requests
import time

def get_citations(paper_id):
    """获取指定论文的所有引用文献ID和标题"""
    citations = {}
    # 使用 Semantic Scholar Graph API
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
    params = {'fields': 'title,year,authors,venue', 'limit': 1000}
    
    try:
        response = requests.get(url, params=params).json()
        if 'data' in response:
            for item in response['data']:
                citing_paper = item.get('citingPaper')
                if citing_paper and citing_paper.get('paperId'):
                    citations[citing_paper['paperId']] = citing_paper
        return citations
    except Exception as e:
        print(f"Error fetching citations: {e}")
        return {}

# 1. 定义两篇论文的 ID (Semantic Scholar ID 或 ArXiv ID)
# A: Measuring Faithfulness in Chain-of-Thought Reasoning (Lanham et al.)
paper_a_id = "arXiv:2307.13702" 

# B: Language Models Don't Always Say What They Think (Turpin et al.)
paper_b_id = "arXiv:2305.04388"

print("正在获取引用列表...")
citations_a = get_citations(paper_a_id)
citations_b = get_citations(paper_b_id)

print(f"引用了 A 的文章数: {len(citations_a)}")
print(f"引用了 B 的文章数: {len(citations_b)}")

# 2. 计算差集 (In A but NOT in B)
unique_ids = set(citations_a.keys()) - set(citations_b.keys())

# 3. 输出结果
print(f"\nFound {len(unique_ids)} papers that cite A but not B:\n")
print("-" * 60)
for pid in unique_ids:
    paper = citations_a[pid]
    title = paper.get('title', 'Unknown Title')
    year = paper.get('year', 'N/A')
    print(f"[{year}] {title}")
    # print(f"Link: https://www.semanticscholar.org/paper/{pid}") # 可选：打印链接

# 生成RIS文件
with open('citations.ris', 'w', encoding='utf-8') as f:
    for pid in unique_ids:
        paper = citations_a[pid]
        f.write('TY  - JOUR\n')
        for author in paper.get('authors', []):
            f.write(f'AU  - {author["name"]}\n')
        f.write(f'TI  - {paper.get("title", "")}\n')
        f.write(f'PY  - {paper.get("year", "")}\n')
        if 'venue' in paper and paper['venue']:
            f.write(f'JO  - {paper["venue"]}\n')
        if 'doi' in paper and paper['doi']:
            f.write(f'DO  - {paper["doi"]}\n')
        f.write('ER  -\n\n')