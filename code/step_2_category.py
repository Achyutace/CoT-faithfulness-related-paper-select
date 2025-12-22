import pandas as pd
import os
import re
import yaml
import json
from tqdm import tqdm

import concurrent.futures
from openai import OpenAI

with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
API_KEY = config['API_KEY']

DATA_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/data/"
api_key = API_KEY
api_base_url = "https://yunwu.ai/v1"


def analyze_paper_dimensions(client, title, abstract):
    """
    单个论文打标函数：分析 Type (White/Black box) 和 Domain
    """
    if not abstract or len(abstract) < 20:
        return {"title": title, "related_score": "Unknown", "type": "Unknown", "domain": "Unknown", "tagging_reasoning": "摘要缺失"}

    system_instruction = """
    你是 NLP 领域专注于 "Explainability" 和 "CoT Faithfulness" 的高级研究员。你需要根据论文的标题和摘要，筛选出适合 "Chain-of-Thought Faithfulness Survey" 的核心文献。
    ### 核心定义
    **Faithfulness (忠实度)** 指的是：模型生成的解释（CoT/Reasoning Trace）是否真实反映了模型做出最终预测的实际计算过程。
    - 反义词：Post-hoc Rationalization（事后合理化）、Sycophancy（阿谀奉承）、Unfaithful。
    - 关键区别：不要把 Faithfulness 混淆为 Correctness（答案正确）或 Factuality（符合客观事实）。

    ### 打分标准
    **[2分] Core Relevant (核心相关)**
    - 论文明确研究 CoT 是否是模型预测的真实原因（Causal role）。
    - 论文研究 CoT 的“欺骗性”或“事后解释”现象（Post-hoc rationalization, Sycophancy）。
    - 论文提出了测量 Faithfulness 的指标（如：Counterfactual input, Early exiting, Feature attribution on CoT）。
    - 关键词：Faithfulness, Unfaithful, Rationalization, Sycophancy, Illusion of reasoning, Causal proxy.

    **[1分] Borderline / Context (边缘相关)**
    - 论文研究 LLM 推理的内部机制（Mechanistic Interpretability），虽未直指 Faithfulness，但分析了 Attention 或 Circuit。
    - 论文研究推理的鲁棒性（Robustness）或一致性（Self-consistency），可以作为 Faithfulness 的旁证。
    - 论文讨论了 CoT 的效用（Utility），即 CoT 到底有没有用，但未深入因果分析。

    **[0分] Irrelevant (不相关 - 必须剔除)**
    - **纯性能提升**：仅提出新 Prompt 策略（如 ToT, GoT, PoT）以提高准确率（SOTA），不关心解释的真实性。
    - **事实性（Factuality）**：仅研究如何减少幻觉（Hallucination）或利用 RAG 增强知识，不涉及推理过程的忠实度。
    - **其他应用**：将 CoT 应用于特定领域（如医疗、法律）但仅作为工具使用。

    此外，你还要对论文做一下分类：
    2. <type>: 方法类型
       - White-box (白盒)：研究涉及模型内部机制、激活值、梯度、权重或电路(Circuits)。
       - Black-box (黑盒)：仅通过输入输出、Prompting 或 API 进行研究。

    3. <domain>: 任务领域
       - Math (数学)
       - Logic (逻辑)
       - Code (代码)
       - Medical (医学)
       - Society (社会学)：涉及偏见、道德、人类行为、社会经济因素等。
       - General (通用)：如果不属于上述特定领域，请填此项。

    

    输出格式要求：
    必须输出包含在 <JSON_OUTPUT> 标签内的 JSON 对象，格式如下：
    <JSON_OUTPUT>
    {
        "related_score": 0 或 1 或 2,
        "type": "White-box 或 Black-box",
        "domain": "上述 6 个分类之一",
        "tagging_reasoning": "简要理由（中文）"
    }
    </JSON_OUTPUT>
    """

    user_content = f"请评估以下论文：Title: {title}\nAbstract: {abstract} 判断它是否属于 CoT Faithfulness 的研究范畴。"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        
        # 使用正则提取标签内容
        match = re.search(r'<JSON_OUTPUT>(.*?)</JSON_OUTPUT>', content, re.DOTALL | re.IGNORECASE)
        if match:
            result = json.loads(match.group(1).strip())
        else:
            # 降级处理
            json_str = content.replace('```json', '').replace('```', '').strip()
            result = json.loads(json_str)
            
        result["title"] = title
        return result
    except Exception as e:
        return {"title": title, "related_score": "Error", "type": "Error", "domain": "Error", "tagging_reasoning": str(e)}

def parallel_tagging(papers, client, max_workers=10):
    """
    并行处理函数
    :param papers: 包含 'title' 和 'abstract' 字典的列表
    """
    results = []
    print(f"开始并行打标，任务总数: {len(papers)}, 线程数: {max_workers}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_paper = {
            executor.submit(analyze_paper_dimensions, client, p['title'], p['abstract']): p 
            for p in papers
        }
        
        # 使用 tqdm 显示进度（注意修复了之前你遇到的 total 传参问题）
        for future in tqdm(concurrent.futures.as_completed(future_to_paper), 
                          total=len(papers), 
                          desc="Tagging Papers"):
            try:
                data = future.result()
                results.append(data)
            except Exception as exc:
                print(f"生成的异常: {exc}")
                
    return results



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

# sort_by_X("has_phenomenon", ["True", "False"])

def main(version = "v1"):
    TAGGING = False  # 是否重新打标
    # TAGGING = True  # 打开此行以重新打标
    if TAGGING == True:
        # 1. 初始化客户端
        client = OpenAI(api_key=api_key, base_url=api_base_url)

        # 2. 准备论文数据列表（仅处理有摘要的行以节省 Token）
        # 将 DataFrame 转换为 parallel_tagging 期待的格式
        with open(DATA_PATH + "first_selected_papers.csv", "r", encoding="utf-8") as f:
            old_df = pd.read_csv(f)
        papers_to_process = old_df[['title', 'abstract']].dropna(subset=['abstract']).to_dict('records')
        
        # 3. 执行并行打标
        # 建议 max_workers 根据你的 API 限制调整，通常 5-10 比较安全
        tagging_results = parallel_tagging(papers_to_process, client, max_workers=20)

        # 4. 将结果转为 DataFrame 并合并回主表
        # 使用 title 作为 key 进行 merge
        results_df = pd.DataFrame(tagging_results)
        
        # 我们用 global df 来接收更新，确保 sort_by_X 能读到新列
        df = pd.merge(old_df, results_df, on='title', how='left')

        # 5. 保存带有打标结果的全量 CSV
        full_output_path = os.path.join(DATA_PATH, f"papers_with_dimensions_{version}.csv")
        df.to_csv(full_output_path, index=False, encoding="utf-8-sig")
        print(f"\n[完成] 全量打标结果已保存至: {full_output_path}")
    
    else:
        # 如果不打标，直接读取已有结果
        df = pd.read_csv(DATA_PATH + f"papers_with_dimensions_{version}.csv")

    # 6. 调用分类函数进行物理分表（自动创建文件夹）
    print("\n[分类] 正在按维度进行物理分表...")
    
    # 按 Type 分类
    sort_by_X("type", ["White-box", "Black-box"], version=version)
    
    # 按 Domain 分类
    sort_by_X("domain", ["Math", "Logic", "Code", "Medical", "Society", "General"], version=version)

    sort_by_X("related_score", [0, 1, 2], version=version)

    sort_by_X("mitigation_methods", ["Training & Fine-tuning", "Interpretability & Internal Mechanisms", "Prompting & In-Context Learning", "Verification & External Tools", "Consistency & Ensembling"], version=version)

    sort_by_X("has_phenomenon", ["True", "False"], version=version)

    sort_by_X("has_metrics", ["True", "False"], version=version)
if __name__ == "__main__":
    main("v3")