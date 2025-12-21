'''
sort_paper 的 Docstring
用于收集并分类了所有引用了两篇核心论文的相关论文
并使用 DeepSeek API 进行多维度分类
核心论文：
- Measuring Faithfulness in Chain-of-Thought Reasoning (Lanham et al.)
- Language Models Don't Always Say What They Think (Turpin et al.)
- Towards Faithfully Interpretable NLP Systems (DeYoung et al.)
分类结果储存在 faithfulness_papers_full_survey.csv 中

TODO: 修改为并行调用API
TODO: json修改为html标签解析, 提升鲁棒性
'''
import os
import json
import time
import yaml
import pandas as pd
from semanticscholar import SemanticScholar
from openai import OpenAI
from tqdm import tqdm
import re

# --- 配置部分 ---
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
API_KEY = config['API_KEY']
DEEPSEEK_API_KEY = API_KEY # 替换为你的 Key
BASE_URL = "https://yunwu.ai/v1"  # 云雾端点

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
sch = SemanticScholar()

# 目标核心论文标题
SEED_PAPERS = [
    "Measuring Faithfulness in Chain-of-Thought Reasoning",
    "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting",
    "Towards Faithfully Interpretable NLP Systems"
]
def extract_tag_content(text, tag_name, default=None):
    """
    辅助函数：使用正则从文本中提取 <tag>内容</tag>
    """
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return default
def analyze_cot_faithfulness_robust(client, title, abstract):
    """
    鲁棒版本的分析函数。
    1. 使用 System Prompt 注入定义。
    2. 强制模型用 <JSON_OUTPUT> 标签包裹内容，便于正则提取，防止 Markdown 干扰。
    """
    
    # 如果摘要太短或为空，直接跳过，节省 Token
    if not abstract or len(abstract) < 50:
        return None

    system_instruction = """
    你是一位专业的 AI 科研助手，正在撰写一篇关于 "Chain-of-Thought (CoT) Faithfulness" 的综述。
    请阅读用户提供的论文标题和摘要，基于以下定义体系进行详细的分类打标。

    ### 一、核心定义 (Definition Context)
    
    **1. Chain-of-Thought (CoT):**
       - 定义：位于 Input 和 Output 之间的一系列自然语言（Natural Language）中间思考步骤。
       - 核心特征：Intermediate (中间性) + Reasoning (对最终结果有逻辑贡献)。
       - 注意：我们只关注那些声称利用了推理过程的模型，而非直接输出答案的任务。

    **2. Faithfulness (忠实性):**
       - 核心定义（基于 Jacovi & Goldberg, 2020）：模型的 CoT 在多大程度上真实反映了模型内部产生最终答案的计算过程？(True internal process vs. Plausibility).
       - 关键区别：
         - **Unfaithful:** 模型给出的解释（CoT）看起来合理（Plausible），但并不是它得出结论的真实原因（例如：Post-hoc Rationalization, Sycophancy）。
         - **Faithful:** 如果删去 CoT 中的关键步骤，模型结论会改变（Counterfactual），或者 CoT 的生成过程与模型内部电路（Circuits）有因果联系。

    ### 二、分类标准 (Classification Schema)

    请判断论文是否涉及以下维度（可多选）：

    **[A] Phenomenon (现象):**
       - 论文是否揭示了不忠实现象？
       - 关键词：Rationalization (事后找补), Sycophancy (阿谀奉承), Inconsistency (推理对答案错/推理错答案对), Shortcut Learning (捷径), Post-hoc explanation.

    **[B] Evaluation Frameworks or Metrics (度量):**
       - 论文是否提出了新的评估框架或具体指标？
       - 黑盒：Consistency checks, Input ablation/perturbation.
       - 白盒：Causal abstraction, Attention attribution, Probing.

    **[C] Mitigation (改进方法):**
       - 论文是否提出了提升 Faithfulness 的方法？如有，请归入以下类别（可多选）：
         1. **Training & Fine-tuning:** RLHF, SFT, Distillation (如 OpenAI o1, LLaVA-CoT).
         2. **Verification & External Tools:** Solvers, Python Interpreters, Verifiers (如 Faithful CoT).
         3. **Prompting & In-Context Learning:** Self-Correction prompts, Rephrase and Respond, Decomposition.
         4. **Interpretability & Internal Mechanisms:** Activation Steering, Latent Space editing (白盒干预).
         5. **Consistency & Ensembling:** Self-Consistency, Voting, Diffusion of Thought.

    ### 三、输出格式 (JSON Output Only)
    请仅输出一个合法的 JSON 对象，不要包含 Markdown 格式（```json ... ```）：
    {
        "is_CoT_relevant": true/false,  // 是否与 Chain-of-Thought 高度相关
        "is_faithfulness_relevant": true/false, // 是否与 Faithfulness 高度相关
        "is_relevant": true/false,  // 是否与 CoT Faithfulness 高度相关
        "has_phenomenon": true/false, // 是否讨论了不忠实现象
        "has_metrics": true/false,    // 是否提出了度量指标/框架
        "mitigation_methods": [],     // 列表，仅包含上述 C 中的 5 个固定分类名称，如果没有则为空列表
        "reasoning": "简要说明判断理由（中文），不超过 50 字。"
    }
    """

    user_content = f"""
    Title: "{title}"
    Abstract: "{abstract}"
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat", # 指向 DeepSeek V3
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            
            # --- 鲁棒性处理核心：正则提取 ---
            # 提取 <JSON_OUTPUT> 标签内的内容，忽略外部的寒暄或 Markdown 符号
            match = re.search(r'<JSON_OUTPUT>(.*?)</JSON_OUTPUT>', content, re.DOTALL)
            
            if match:
                json_str = match.group(1).strip()
            else:
                # 降级处理：尝试直接去除 Markdown 代码块
                json_str = content.replace('```json', '').replace('```', '').strip()

            result_json = json.loads(json_str)
            return result_json

        except json.JSONDecodeError:
            print(f"JSON Decode Error for: {title} (Attempt {attempt+1})")
        except Exception as e:
            print(f"API Error: {e} (Attempt {attempt+1})")
            time.sleep(1) # 简单的 backoff
    
    return None



def fetch_citations_from_s2(paper_titles):
    """
    从 Semantic Scholar 获取引用了指定论文的所有文献
    """
    all_citations = []
    seen_ids = set()
    
    print("正在从 Semantic Scholar 获取引用数据...")
    
    for seed_title in paper_titles:
        print(f"正在搜索种子论文: {seed_title}")
        results = sch.search_paper(seed_title, limit=1)
        
        if not results:
            print(f"未找到论文: {seed_title}")
            continue
            
        seed_paper = results[0]
        print(f"找到种子论文 (ID: {seed_paper.paperId}), 开始获取引用...")
        
        # 获取引用文献 (Semantic Scholar API 可能需要分页，这里使用库的简便方法)
        # 注意：大量引用可能导致速度较慢，建议限制数量或分批
        citations = sch.get_paper(seed_paper.paperId).citations
        
        if citations:
            for cite in citations:
                if cite.paperId not in seen_ids:
                    # 获取引用论文的详细信息（需要 title 和 abstract）
                    # 这里的 cite 对象可能只有基本信息，需要再次获取详情
                    # 为了演示效率，如果 cite 中已有 abstract 则直接用，否则需额外请求
                    # 实际生产中建议在此处做 Batch 请求优化
                    
                    # 简单起见，我们只添加有标题和摘要的
                    if cite.title and cite.abstract:
                        all_citations.append({
                            "paper_id": cite.paperId,
                            "title": cite.title,
                            "abstract": cite.abstract,
                            "year": cite.year,
                            "citation_count": cite.citationCount,
                            "publication_venue": cite.publicationVenue
                        })
                        seen_ids.add(cite.paperId)
        
    print(f"共获取到 {len(all_citations)} 篇去重后的引用文献。")
    return all_citations

def fetch_citations_with_cache(paper_titles, cache_file="citation_cache.json"):
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            citation_cache = json.load(f)
            return citation_cache

    paper = fetch_citations_from_s2(paper_titles)

    with open (cache_file, "w", encoding="utf-8") as f:
        json.dump(paper, f, ensure_ascii=False, indent=2)
    
    return paper

def main():
    # 1. 获取数据
    papers = fetch_citations_with_cache(SEED_PAPERS, cache_file="citation_cache.json")
    last_select_f = pd.read_csv("data/cot_faithfulness_citations_analysis.csv")
    # 创造true列表，包含所有is_relevant为true的title
    true_titles = last_select_f[last_select_f['is_relevant'] == True]['title'].tolist()
    # 调试模式：只跑前 5 篇测试
    # papers = papers[:5] 
    
    results_data = []
    filename = "cot_faithfulness_citations_analysis.csv"
    

    print("开始调用 DeepSeek V3 进行深度分析...")
    for i, paper in enumerate(tqdm(papers)):
        # 看之前的结果是不是true，如果不是true直接跳过
        if paper['title'] not in true_titles:
            continue
        # 调用 API
        analysis = analyze_cot_faithfulness_robust(
            client, 
            paper['title'], 
            paper['abstract']
        )
        
        if analysis:
            # 合并原始信息和分析结果
            row = {
                "title": paper['title'],
                "year": paper['year'],
                "is_CoT_relevant": analysis.get("is_CoT_relevant"),
                "is_faithfulness_relevant": analysis.get("is_faithfulness_relevant"),
                "is_relevant": analysis.get("is_relevant"),
                "has_phenomenon": analysis.get("has_phenomenon"),
                "has_metrics": analysis.get("has_metrics"),
                # 将列表转为字符串以便存入 CSV
                "mitigation_methods": ", ".join(analysis.get("mitigation_methods", [])),
                "reasoning": analysis.get("reasoning"),
                "abstract": paper['abstract'], # 保留摘要备查
                "publication_venue": paper['publication_venue']
            }
            results_data.append(row)
            
        # 每处理10篇论文自动保存
        if (i + 1) % 10 == 0 and results_data:
            df = pd.DataFrame(results_data)
            # 按照相关性排序（True 在前）
            df = df.sort_values(by="is_relevant", ascending=False)
            df.to_csv(filename, index=False, encoding="utf-8-sig")
            print(f"已处理 {i + 1} 篇论文，自动保存至 {filename}")
            
    # 2. 最终存入 CSV（处理剩余的论文）
    if results_data:
        df = pd.DataFrame(results_data)
        
        # 按照相关性排序（True 在前）
        df = df.sort_values(by="is_relevant", ascending=False)
        
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"分析完成！结果已保存至 {filename}")
        print(f"其中高度相关论文数量: {df['is_relevant'].sum()}")
    else:
        print("未获取到有效数据。")

if __name__ == "__main__":
    main()