'''
sort_paper 的 Docstring
用于收集并分类了所有引用了两篇核心论文的相关论文
并使用 DeepSeek API 进行多维度分类
核心论文：
- Measuring Faithfulness in Chain-of-Thought Reasoning (Lanham et al.)
- Language Models Don't Always Say What They Think (Turpin et al.)
分类结果储存在 faithfulness_papers_full_survey.csv 中
'''
import requests
import pandas as pd
import json # 这里的json库仅用于其他用途，解析不再依赖它
import time
import yaml
import re   # 引入正则表达式库
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 请务必替换为你的 Key
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
API_KEY = config['API_KEY']
DEEPSEEK_API_KEY = API_KEY

# 2. API Base URL
BASE_URL = "https://yunwu.ai/v1"

# 3. 核心种子论文 ID
SEED_PAPER_IDS = ["arXiv:2307.13702", "arXiv:2305.04388"]

# 4. 输出文件名
OUTPUT_FILE = "faithfulness_papers_full_survey.csv"
# ===========================================

def get_citations_from_semantic_scholar(paper_id):
    """
    从 Semantic Scholar 获取引用了指定论文的所有文章列表
    """
    print(f"正在获取引用了 {paper_id} 的论文列表...")
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
    params = {'fields': 'title,abstract,year,citationCount,authors', 'limit': 1000}
    
    try:
        response = requests.get(url, params=params).json()
        papers = []
        if 'data' in response:
            for item in response['data']:
                citing_paper = item.get('citingPaper')
                if citing_paper and citing_paper.get('abstract') and citing_paper.get('title'):
                    papers.append({
                        'paperId': citing_paper.get('paperId'),
                        'title': citing_paper.get('title'),
                        'abstract': citing_paper.get('abstract'),
                        'year': citing_paper.get('year'),
                        'citations': citing_paper.get('citationCount')
                    })
        print(f" -> 找到 {len(papers)} 篇有效引用。")
        return papers
    except Exception as e:
        print(f"获取论文列表失败: {e}")
        return []

def extract_content(text, tag):
    """
    鲁棒性核心：使用正则提取 <tag>...</tag> 之间的内容
    re.DOTALL 允许匹配跨行内容
    re.IGNORECASE 允许忽略大小写
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Unknown" # 如果没找到标签，返回 Unknown

def classify_with_deepseek(client, title, abstract):
    """
    调用 DeepSeek API 进行多维度详细分类 (使用 HTML Tags 模式)
    """
    prompt = f"""
    你是一位专业的 AI 科研助手。请根据以下论文的标题和摘要，分析其关于思维链（CoT）忠实度（Faithfulness）的内容。
    
    论文标题: "{title}"
    摘要内容: "{abstract}"
    
    请提取以下 5 个维度的关键信息，并严格包裹在 XML/HTML 标签中输出：

    1. <category>: 主要类别
       - Phenomenon (现象发现)
       - Metric (评估指标)
       - Mitigation (改进方法)
       - Other (其他)

    2. <type>: 方法类型
       - White-box (白盒)
       - Black-box (黑盒)

    3. <domain>: 任务领域
       - Math (数学)
       - Logic (逻辑)
       - Code (代码)
       - General (通用)

    4. <tradeoff>: 性能权衡 (仅针对 Mitigation)
       - Positive (双赢)
       - Negative (牺牲准确率)
       - Unknown (未知)

    5. <cost>: 推理开销
       - High (高成本，如多次采样)
       - Low (低成本，如单次推理)
       
    6. <reasoning>: 简短中文理由 (30字以内)

    输出示例模板 (不要输出 Markdown 代码块，直接输出以下文本):
    <category>Mitigation</category>
    <type>Black-box</type>
    <domain>Math</domain>
    <tradeoff>Positive</tradeoff>
    <cost>Low</cost>
    <reasoning>这篇论文提出了一种新的Prompt策略。</reasoning>
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."}, # 不再强制 JSON
                {"role": "user", "content": prompt}
            ],
            # response_format={ "type": "json_object" }, # <--- 移除这一行，这是关键！
            temperature=0.0
        )
        
        raw_content = response.choices[0].message.content
        
        # 使用正则解析结果，这比 json.loads 鲁棒得多
        return {
            "category": extract_content(raw_content, "category"),
            "method_type": extract_content(raw_content, "type"),
            "task_domain": extract_content(raw_content, "domain"),
            "tradeoff": extract_content(raw_content, "tradeoff"),
            "cost": extract_content(raw_content, "cost"),
            "reasoning": extract_content(raw_content, "reasoning")
        }

    except Exception as e:
        print(f"API Error: {e}")
        return {
            "category": "Error", "method_type": "Error", 
            "task_domain": "Error", "tradeoff": "Error", "cost": "Error", 
            "reasoning": str(e)
        }

def retry_failed_rows(csv_file):
    """
    读取 CSV 文件，查找分类失败（Reasoning 中包含错误信息或 Category 为 Error）的行，
    并重新调用 API 进行分类，最后更新文件。
    """
    print(f"🔄 正在检查 {csv_file} 中的失败项...")
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("❌ 未找到文件，请先运行主程序收集数据。")
        return

    # 1. 定义失败的条件
    # 条件A: Category 被标记为 'Error' (根据你之前的异常处理逻辑)
    # 条件B: Reasoning 中包含 'time out' 或 'timed out' (不区分大小写)
    # 条件C: Category 是 'Unknown' (解析失败)
    error_mask = (
        (df['Category'] == 'Error') | 
        (df['Category'] == 'Unknown') |
        (df['Reasoning'].astype(str).str.contains('time out', case=False, regex=True)) |
        (df['Reasoning'].astype(str).str.contains('Error', case=False, regex=True))
    )
    
    failed_rows = df[error_mask]
    
    if failed_rows.empty:
        print("✅ 没有发现失败的行，无需重试。")
        return

    print(f"⚠️ 发现 {len(failed_rows)} 个失败或超时的条目，开始重试...\n")
    
    # 初始化 API Client (确保 Key 正确)
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
    
    # 2. 遍历失败的行进行重试
    # 使用 .index 获取原始行号，确保直接修改原 DataFrame 的对应位置
    for idx in tqdm(failed_rows.index, desc="Retrying"):
        row = df.loc[idx]
        title = row['Title']
        abstract = row['Abstract']
        
        # 重新调用分类函数
        analysis = classify_with_deepseek(client, title, abstract)
        
        # 3. 只有当成功（不是 Error 且不是 Unknown）时才更新
        # 如果这次又失败了，保留原来的错误信息或者更新为新的错误信息皆可
        if analysis['category'] != 'Error':
            df.at[idx, 'Category'] = analysis['category']
            df.at[idx, 'Type'] = analysis['method_type']
            df.at[idx, 'Domain'] = analysis['task_domain']
            df.at[idx, 'Tradeoff'] = analysis['tradeoff']
            df.at[idx, 'Cost'] = analysis['cost']
            df.at[idx, 'Reasoning'] = analysis['reasoning']
        else:
            # 如果又失败了，更新一下错误原因（可能是不同的错误）
            df.at[idx, 'Reasoning'] = analysis['reasoning']

        # 每次处理完稍微停顿，防止再次触发限流
        time.sleep(0.5) 
        
        # 每重试 5 个保存一次，防止程序中断白跑
        if idx % 5 == 0:
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')

    # 4. 最终保存
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    # 统计修复情况
    remaining_errors = df[
        (df['Category'] == 'Error') | 
        (df['Reasoning'].astype(str).str.contains('time out', case=False))
    ].shape[0]
    
    print(f"\n🎉 重试结束！")
    print(f"原始失败数: {len(failed_rows)}")
    print(f"剩余失败数: {remaining_errors}")
    print(f"成功修复数: {len(failed_rows) - remaining_errors}")

def main():
    # 1. 初始化

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
    
    all_results = []
    seen_titles = set()
    papers_to_process = []

    # === Step 1: 收集 ===
    print("Step 1: 正在收集所有引用论文...")
    for seed_id in SEED_PAPER_IDS:
        fetched_papers = get_citations_from_semantic_scholar(seed_id)
        for paper in fetched_papers:
            if paper['title'] not in seen_titles:
                seen_titles.add(paper['title'])
                papers_to_process.append(paper)
    
    print(f"\n去重后，共需处理 {len(papers_to_process)} 篇论文。")
    if not papers_to_process: return

    # === Step 2: 分类 ===
    print("\nStep 2: 开始 AI 智能打标 (Regex Robust Mode)...")
    
    for paper in tqdm(papers_to_process): # 测试时可加 [:5]
        
        analysis = classify_with_deepseek(client, paper['title'], paper['abstract'])
        
        entry = {
            "Title": paper['title'],
            "Year": paper['year'],
            "Citations": paper['citations'],
            # --- 详细指标 ---
            "Category": analysis.get('category'),
            "Type": analysis.get('method_type'),
            "Domain": analysis.get('task_domain'),
            "Tradeoff": analysis.get('tradeoff'),
            "Cost": analysis.get('cost'),
            "Reasoning": analysis.get('reasoning'),
            # ---------------
            "Abstract": paper['abstract']
        }
        all_results.append(entry)
        
        if len(all_results) % 10 == 0:
            pd.DataFrame(all_results).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
        time.sleep(0.1)

    # === Step 3: 保存 ===
    df = pd.DataFrame(all_results)
    df = df.sort_values(by="Citations", ascending=False)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 全部完成！结果已保存至: {OUTPUT_FILE}")
    if 'Category' in df.columns:
        print("📊 分类统计预览:")
        print(df['Category'].value_counts())

if __name__ == "__main__":
    mode = "retry"
    if mode == "retry":
        retry_failed_rows(OUTPUT_FILE)
    else:
        main()
