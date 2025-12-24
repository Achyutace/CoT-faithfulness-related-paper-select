import os
import glob
import pandas as pd
import pdfplumber
import json
import yaml
import re
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
# 文件路径 (注意 Windows 路径需要转义或用 rString)
META_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = META_PATH + "/data/"
CSV_PATH = DATA_PATH + "filtered_faithfulness_papers_v1.csv"
PDF_DIR = DATA_PATH + "papers"
OUTPUT_CSV_PATH = DATA_PATH + "screened_papers_result.csv"

# 替换为你的 API Key 和 Base URL
with open(META_PATH + "/config.yaml", "r") as f:
    config = yaml.safe_load(f)
API_KEY = config['api_key']
BASE_URL = "https://yunwu.ai/v1" # 如果用别的模型（如DeepSeek, Claude等），请修改此处
MODEL_NAME = "gpt-4o" # 建议使用能力较强的模型


# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def clean_filename(title):
    """
    将所有特殊字符（包括冒号、问号等）替换为下划线。
    只保留字母、数字、空格和下划线。
    """
    if not isinstance(title, str):
        return ""
    # 将所有非字母数字、非空格、非下划线的字符替换为下划线
    cleaned = re.sub(r'[^\w\s]', "_", title)
    # 压缩多个下划线或空格为单个下划线
    cleaned = re.sub(r'[_\s]+', "_", cleaned)
    # 去掉首尾下划线
    cleaned = cleaned.strip("_")
    # 限制长度
    if len(cleaned) > 150:
        cleaned = cleaned[:150].rstrip("_")
    return cleaned

def extract_text_from_pdf(pdf_path, max_pages=3):
    """
    提取 PDF 前几页的文本。
    通常 Abstract, Intro, Method 在前 3 页就足够判断了。
    """
    text_content = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= max_pages:
                    break
                text = page.extract_text()
                if text:
                    text_content += text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None
    return text_content

def check_paper_with_llm(paper_text):
    """
    调用 LLM 进行判断
    """
    system_prompt = """
   你是一位自然语言处理（NLP）和可解释性（Interpretability）领域的专家学者，目前正在撰写一篇关于**“思维链忠实度 (CoT Faithfulness)”**的综述论文。

你的任务是阅读输入的论文片段（通常是摘要、引言和方法），判断该论文是否应该被**收录**。

请基于以下**严格标准**进行筛选：

### 排除标准 (Reject Criteria) - 满足任意一条即排除：
1.  **非 CoT (Non-CoT)**：该方法完全不涉及 Chain-of-Thought 或逐步推理（Step-by-step reasoning）机制。
2.  **纯事后解释 (Post-hoc Only)**：这是最关键的过滤点。如果论文使用的是“Self-Explanation”或其他可解释性方法，但其解释是**事后（Post-hoc）**生成的（例如：先输出标签，再生成解释；或者解释是由一个独立模块生成，不参与最终预测的计算），必须排除。我们需要的是 Intrinsic（内在）的推理，即推理过程决定了预测结果。
3.  **无关内容 (Irrelevant)**：论文与大语言模型（LLM）、推理或可解释性完全无关。

### 保留标准 (Keep Criteria)：
- 论文讨论了 CoT 的 Faithfulness（忠实度）、Fidelity（保真度）问题（即：CoT 文本是否真实反映了模型的底层推理过程）。
- 论文提出了一种 Intrinsic（内在）的 CoT 方法，旨在提高推理的忠实度。
- 论文分析了 CoT 推理过程中的某种机制。

### 输出格式：
请必须输出合法的 JSON 格式，包含以下字段：
- `is_cot`: (boolean) 论文是否涉及思维链或逐步推理？
- `is_post_hoc_only`: (boolean) 该方法是否属于纯粹的事后解释（即解释不影响预测结果）？
- `decision`: (string) 输出 "KEEP" 或 "REJECT"。
- `reason`: (string) 请用**中文**简要说明理由。如果拒绝，请明确指出是因为“非CoT”还是“事后解释”。"""
    user_prompt = f"Paper Content (First few pages):\n\n{paper_text[:8000]}" # 截断以防超出 token 限制

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"} # 强制 JSON 输出
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"API Error: {e}")
        return {"is_cot": "ERROR", "is_post_hoc_only": "ERROR", "decision": "ERROR", "reason": str(e)}

def main():
    # 1. 读取 CSV
    df = pd.read_csv(CSV_PATH)
    
    # 初始化结果列（如果还没有）
    if 'is_cot' not in df.columns:
        df['is_cot'] = ""
        df['is_post_hoc_only'] = ""
        df['decision'] = ""
        df['reason'] = ""
        df['pdf_found'] = False

    # 获取所有 PDF 文件列表，方便匹配
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    # 构建一个 {清洗后的文件名: 完整路径} 的映射，提高查找效率
    pdf_map = {os.path.basename(p).lower(): p for p in pdf_files}

    print(f"Found {len(pdf_files)} PDFs in directory.")

    # 2. 遍历每一行
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # 如果已经跑过且成功，跳过（方便断点续传）
        if pd.notna(row.get('decision')) and row['decision'] != "":
            continue

        title = row['title']
        clean_title = clean_filename(title)
        
        # 尝试匹配 PDF
        # PDF 文件名规则：标题清洗后（所有特殊字符替换为下划线）+ .pdf
        # 尝试 1: 精确匹配清洗后的标题 + .pdf
        target_pdf_name = clean_title.lower() + ".pdf"
        pdf_path = pdf_map.get(target_pdf_name)

        # 尝试 2: 如果找不到，尝试简单的包含匹配（匹配前30个字符，去掉下划线后比较）
        if not pdf_path:
            clean_title_normalized = re.sub(r'_+', '_', clean_title.lower())
            for p_name, p_path in pdf_map.items():
                p_name_normalized = re.sub(r'_+', '_', p_name.replace('.pdf', ''))
                if clean_title_normalized[:30] in p_name_normalized or p_name_normalized[:30] in clean_title_normalized:
                    pdf_path = p_path
                    break
        
        if pdf_path:
            df.at[index, 'pdf_found'] = True
            # 提取文本
            text = extract_text_from_pdf(pdf_path)
            if text and len(text) > 500: # 确保提取到了有效内容
                # LLM 判断
                result = check_paper_with_llm(text)
                df.at[index, 'is_cot'] = result.get('is_cot', 'ERROR')
                df.at[index, 'is_post_hoc_only'] = result.get('is_post_hoc_only', 'ERROR')
                df.at[index, 'decision'] = result.get('decision', 'ERROR')
                df.at[index, 'reason'] = result.get('reason', 'No reason provided')
            else:
                df.at[index, 'is_cot'] = "ERROR_TEXT"
                df.at[index, 'is_post_hoc_only'] = "ERROR_TEXT"
                df.at[index, 'decision'] = "ERROR_TEXT"
                df.at[index, 'reason'] = "Text extraction failed or too short"
        else:
            df.at[index, 'pdf_found'] = False
            df.at[index, 'is_cot'] = "PDF_MISSING"
            df.at[index, 'is_post_hoc_only'] = "PDF_MISSING"
            df.at[index, 'decision'] = "PDF_MISSING"
            df.at[index, 'reason'] = "Could not find corresponding PDF file"

        # 每处理 5 篇保存一次，防止崩溃
        if index % 5 == 0:
            df.to_csv(OUTPUT_CSV_PATH, index=False)

    # 最后保存
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Done! Results saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()