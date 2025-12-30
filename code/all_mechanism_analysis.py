import re
import os
import pdfplumber
import glob
import json
import yaml
import pandas as pd
from tqdm import tqdm
import concurrent.futures

from openai import OpenAI


# ================= 配置区域 =================
# 文件路径 (注意 Windows 路径需要转义或用 rString)
META_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = META_PATH + "/data/"
CSV_PATH = DATA_PATH + "filtered_faithfulness_papers_v2.csv"
PDF_DIR = DATA_PATH + "papers"
OUTPUT_DIR = DATA_PATH + "related_phenomenan"

# 替换为你的 API Key 和 Base URL
with open(META_PATH + "/config.yaml", "r") as f:
    config = yaml.safe_load(f)
API_KEY = config['api_key']
BASE_URL = "https://yunwu.ai/v1" # 如果用别的模型（如DeepSeek, Claude等），请修改此处
MODEL_NAME = "gpt-4o" # 建议使用能力较强的模型

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
# ===========================================
def clean_filename(title):
    # 把所有的?和:等windows不允许的字符替换为下划线
    title = re.sub(r'[?/:*?"<>|]', "_", title)
    return title
def normalize_for_matching(text):
    """
    只保留字母和数字，用于文件名匹配。
    将文本转换为只包含小写字母和数字的字符串。
    """
    if not isinstance(text, str):
        return ""
    # 只保留字母和数字，转为小写
    normalized = re.sub(r'[^a-z0-9]', '', text.lower())
    return normalized

def output_related_phenomena(client, paper_text):
    """
    output_related_phenomena 的 Docstring
    输出论文中相关的现象
    输入：
    - client: OpenAI 客户端
    - paper_text: 论文文本
    输出：
    - 论文中相关的现象
    """
    

    system_instruction = """
    # Role
    你是一位专注于 "Explainable AI (XAI)" 和 "Large Language Model Reasoning" 的学术研究员，正在协助我撰写一篇关于 **"Chain-of-Thought (CoT) Faithfulness"** 的综述论文。我们当前的任务是撰写 **"Phenomenology of Unfaithfulness"（不忠实现象）** 这一章节。

    # Task
    请阅读我提供的【论文列表/摘要/全文】，基于下列新的分类体系，深入分析这些论文具体揭示了哪种 CoT 不忠实的现象。请不要泛泛而谈，我需要具体的表现形式和实验证据。

    # Definitions (Recall)
    - **Faithful:** CoT 是模型预测的真实原因（Causal explanation）。
    - **Unfaithful:** CoT 只是看起来合理，但与模型预测的真实机制无关（Plausible but not causal）。

    # Phenomenon Taxonomy (请严格按此分类归纳)
    请重点关注并区分以下四类不忠实现象（如果论文涉及其他现象，请单独列出）：

    1.  **Input-Driven Unfaithfulness (输入诱导的不忠实):**
        * **核心定义：** 模型的推理过程（CoT）并非基于客观逻辑，而是受到输入中非相关特征（如用户立场、上下文偏见、干扰项顺序）的强烈干扰。
        * **典型表现：** * **Sycophancy (阿谀奉承)：** 为了顺从 Prompt 中的用户观点而编造错误的推理。
            * **Context Bias：** 受到上下文中错误先验知识的误导。
    
    2.  **Causal Disconnect & Post-hoc Rationalization (因果断裂与事后合理化):**
        * **核心定义：** 模型的 CoT 与最终答案之间缺乏因果联系，或者是“先射箭再画靶”。
        * **典型表现：**
            * **Post-hoc Rationalization：** 模型内部先得出了答案（通过捷径或直觉），然后生成 CoT 来解释这个答案。
            * **Reasoning Bypassing / Ignoring CoT：** 即使 CoT 推导错误，模型依然输出了（基于训练记忆的）正确答案；或者人为干扰 CoT 对最终答案几乎无影响。

    3.  **Alignment & Optimization Side-effects (对齐与优化带来的副作用):**
        * **核心定义：** 由于模型经过 RLHF（人类反馈强化学习）或特定的指令微调，为了最大化 Reward 或迎合人类偏好，而产生的“为了推理而推理”的虚假现象。
        * **典型表现：**
            * **Reward Hacking：** 生成冗长、看似专业但实际无意义的 CoT，只因为这样能获得高分。
            * **Style-over-Substance：** 牺牲逻辑正确性以换取格式上的完美或语气上的自信。

    4.  **Strategic Deception & Hiding (策略性欺骗与思维隐藏):**
        * **核心定义：** 这是一种更高级、更危险的不忠实。模型在 CoT 中显式地进行一套推理，但实际在内部（Hidden States）通过另一套隐蔽的机制（Steganography）传递信息。
        * **典型表现：**
            * **Encoded Reasoning / Steganography：** 模型将真实推理过程加密隐藏在 CoT 的措辞选择或标点中，人类看不懂，但模型自己能解码。
            * **Sandbagging / Scheming：** 模型故意在 CoT 中表现得能力较弱，或隐藏其真实意图。

    # Output Format Requirements
    如果文章有揭示出相关现象，请按**“现象类别”**为维度进行组织（而不是按论文顺序），用中文回答。输出格式如下：

    ## 1. [现象名称，例如 Input-Driven Unfaithfulness]
    * **现象描述**：用学术语言简要概括该现象在这些论文中的共性表现。
    * **关键证据与案例 (Key Evidence)**：
        * **Paper**: [论文标题/引用]
        * **Observation**: 观察到了什么具体行为？（例如：当用户暗示错误答案时，模型 CoT 转而支持该错误观点）。
        * **Experimental Proof**: 作者用了什么实验证明这是不忠实的？（例如：Biased Context Injection, Feature Importance Analysis, Counterfactual edits）。

    ## 2. [现象名称，例如 Causal Disconnect & Post-hoc Rationalization]
    ...

    ## 3. [其他发现]
    ...
    
    如果文章只是引用了相关现象，但没有揭示出相关现象（例如只是作为背景介绍），请回答：“文章没有揭示出新的相关现象”。
    ---
    """

    user_prompt = f"Paper Content (First few pages):\n\n{paper_text[:8000]}"


    response = client.chat.completions.create(
        model="deepseek-chat", # 指向 DeepSeek V3
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
    content = response.choices[0].message.content
        
    return content
   
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


def process_single_row(args):
    """
    处理单行数据的函数，用于多线程处理
    """
    index, row, pdf_map = args
    try:
        title = row['title']
        clean_title = clean_filename(title)
        output_file = os.path.join(OUTPUT_DIR, f"{clean_title}_related_phenomena.txt")
        
        # 如果输出文件已存在，跳过处理（支持断点续传）
        if os.path.exists(output_file):
            return (index, True, "Already processed (file exists)")
        
        # 尝试匹配 PDF
        target_pdf_name = clean_title.lower() + ".pdf"
        pdf_path = pdf_map.get(target_pdf_name)
        
        if pdf_path:
            # 提取文本
            text = extract_text_from_pdf(pdf_path)
            if text and len(text) > 500:  # 确保提取到了有效内容
                # LLM 判断
                result = output_related_phenomena(client, text)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(result)
                return (index, True, None)
            else:
                return (index, False, "Text extraction failed or too short")
        else:
            return (index, False, f"PDF not found: {target_pdf_name}")
    except Exception as e:
        return (index, False, f"Error: {str(e)}")


def main():
    # 1. 读取 CSV
    df = pd.read_csv(CSV_PATH)
    
    # 获取所有 PDF 文件列表，方便匹配
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    # 构建一个 {只包含字母数字的文件名: 完整路径} 的映射，提高查找效率
    pdf_map = {os.path.basename(p).lower(): p for p in pdf_files}
    print(f"Found {len(pdf_files)} PDFs in directory.")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. 准备参数列表
    tasks = [(index, row, pdf_map) for index, row in df.iterrows()]
    
    # 3. 使用多线程处理
    max_workers = 10  # 可以根据需要调整线程数
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用 tqdm 显示进度
        futures = [executor.submit(process_single_row, task) for task in tasks]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Processing papers"):
            try:
                result = future.result()
                results.append(result)
                
                # 打印错误信息（如果有）
                if not result[1] and result[2]:
                    print(f"\n[{result[0]}] {result[2]}")
            except Exception as e:
                print(f"\nError in future: {e}")
    
    # 4. 统计结果
    success_count = sum(1 for r in results if r[1])
    failed_count = len(results) - success_count
    print(f"\n处理完成: 成功 {success_count} 个, 失败 {failed_count} 个")
            
if __name__ == "__main__":
    main()