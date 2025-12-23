import os
import time
import json
import requests
from typing import Optional

# === 新版 SDK 引入方式 ===
from google import genai
from google.genai import types

# 1. 配置 API Key
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY" # 替换你的 Key
S2_API_KEY = None # 如果有 Semantic Scholar Key 填在这里

# 2. 初始化客户端 (新版用法)
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# === 路径修复 ===
try:
    # 确保脚本作为文件运行时能获取路径，如果是交互式环境(jupyter)可能需要硬编码
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # print(BASE_DIR)
except NameError:
    BASE_DIR = os.getcwd()

DATA_PATH = os.path.join(BASE_DIR, "papers_data") # 示例下载目录

# ================= 核心逻辑 =================

def search_semantic_scholar(title: str) -> Optional[str]:
    """通过 Semantic Scholar 获取 PDF 链接 (保持不变)"""
    print(f"🔍 [S2] 搜索: {title}")
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": title, "limit": 1, "fields": "title,openAccessPdf"}
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}

    try:
        resp = requests.get(base_url, params=params, headers=headers)
        if not resp.json().get("data"): return None
        
        paper = resp.json()["data"][0]
        pdf_info = paper.get("openAccessPdf")
        
        if pdf_info and pdf_info.get("url"):
            return pdf_info["url"]
        print("None")
        return None
    except Exception as e:
        print(f"❌ 搜索出错: {e}")
        return None

def download_pdf(url: str, title: str) -> Optional[str]:
    """下载 PDF"""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print("already make dirs")
        
    safe_title = "".join([c for c in title if c.isalnum() or c in " ._-"]).strip()
    filename = f"{safe_title}.pdf"
    file_path = os.path.join(DATA_PATH, filename)

    if os.path.exists(file_path):
        return file_path

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, stream=True, timeout=30)
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return file_path
    except Exception as e:
        print(f"❌ 下载出错: {e}")
        return None

def analyze_paper_v2(file_path: str) -> str:
    """
    使用新版 google-genai SDK 进行文件上传和分析
    """
    print(f"🚀 [Gemini] 正在上传: {os.path.basename(file_path)}")
    
    # --- 新版上传写法 ---
    # 使用 client.files.upload
    with open(file_path, "rb") as f:
        file_ref = client.files.upload(file=f)
    
    # 等待处理 (新版不再需要手动写 while 循环查询状态，SDK 内部优化了，
    # 但为了保险起见，如果文件很大，可以用 client.files.get 检查 state)
    while file_ref.state.name == "PROCESSING":
        time.sleep(1)
        file_ref = client.files.get(name=file_ref.name)
        
    if file_ref.state.name == "FAILED":
        raise ValueError("文件处理失败")
        
    print("🧠 AI 正在分析...")

    prompt = """
    没问题，为了让你们阅读和整理效率最大化，我把输出要求全部改为中文，同时保留关键术语的英文原词（方便你们引用和检索）。

请使用下面这版 Prompt，直接发给 AI（Claude/GPT-4o），它会变身为一个**“中文综述领读员”**：

📋 专用 Prompt：CoT Unfaithfulness 病理分析（中文版）
【角色设定】 你是一位严苛的 NLP 会议审稿人，正在协助我撰写一篇关于 Chain-of-Thought (CoT) Faithfulness（思维链忠实度） 的综述论文。 你的任务是深入分析我上传的论文，专门挖掘其中提到的 “不忠实现象 (Unfaithful Phenomena)”（即模型的推理过程与真实决策不一致，或存在欺骗/伪造逻辑的情况）。

【阅读指令】

忽略吹嘘： 跳过 Abstract 和 Introduction 中作者对自己模型性能的自夸。

寻找痛点： 重点阅读 Motivation, Problem Definition, 和 Error Analysis 部分。

逆向分析： 如果这是一篇提出新方法的论文，请详细描述它到底是为了解决什么具体的“不忠实”问题而提出的。

【输出格式（请严格使用中文回答）】

请按照以下 Markdown 格式输出分析结果：

1. 核心不忠实现象 (The Phenomenon)
现象名称： 给这个现象起一个简短的中文标签，并附带英文术语（例如：事后合理化 / Post-hoc Rationalization，谄媚 / Sycophancy，逻辑跳跃 / Logical Gap）。

机制定义： 用通俗的语言解释在这里 CoT 是怎么“撒谎”或“失效”的？（例如：“模型先根据偏见猜出了答案，然后编造了一段虚假的推理过程来凑这个答案。”）

严重程度： 这是轻微的逻辑错误，还是完全的推理与答案脱节？

2. 触发场景与领域 (Context & Domain)
触发条件： 在什么情况下容易出现这种不忠实？（例如：“当用户输入带有误导性提示时”、“当问题涉及长文本检索时”、“当答案选项分布不均衡时”）。

所属领域： 论文主要在哪个领域研究此问题？（数学、常识推理、医学、社会科学、代码等）。

3. 具体表现/案例 (Manifestation)
流程复现： 请根据论文内容，构想或摘录一个具体的 Input -> CoT -> Output 错误案例。

用户输入： ...

模型内心/真实倾向（如有）： ...

模型生成的虚假 CoT： ...

验证证据： 作者是如何证明这是不忠实的？（例如：“作者通过干扰输入发现 CoT 变了但答案没变”、“作者使用了线性探针发现答案在第一层就确定了”）。

4. 综述归类标签 (Taxonomy Tags)
请判断这篇论文主要解决哪类问题（可多选，打勾 [x]）：

[ ] 逻辑有效性问题 (Validity/Spuriousness): 解决“过程不对但答案蒙对了”或“逻辑断层”的问题。

[ ] 诚实性与对齐问题 (Honesty/Sycophancy): 解决“谄媚用户”、“欺骗”、“事后合理化”的问题。

[ ] 透明度与可监控性问题 (Transparency/Grounding): 解决“黑盒难懂”、“引用虚假证据”、“人类无法验证”的问题。

注意： 请保持客观、批判的语气。如果论文只是单纯刷榜而没有深入分析 Faithfulness 的机理，请直接指出“本文缺乏对不忠实机理的深入分析”。
    """

    # --- 新版生成写法 ---
    # 1. 只有 model, contents, config 三个主要参数
    # 2. config 使用 types.GenerateContentConfig 封装
    response = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=[file_ref, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
    )
    
    return response.text

# ================= 运行 =================
if __name__ == "__main__":
    target_title = "Attention Is All You Need"
    
    
if __name__ == "__main__":
    # 示例：可以放入一个列表循环处理
    import os
    import pandas as pd
    DATA_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(DATA_PATH)
    
    df = pd.read_csv(f"{DATA_PATH}/DATA/sort_by_has_phenomenon/1faithfulness_papers_true_v1.csv")
    titles = df['title'].tolist()
    
    
    for title in titles:
        pdf_url = search_semantic_scholar(title)
        if pdf_url:
            local_path = download_pdf(pdf_url, title)
            if local_path:
                res = analyze_paper_v2(local_path)
                # 保存为对应标题.markdown
                with open(f"{DATA_PATH}/{title}.md", "w+", encoding="utf-8") as f:
                    f.write(res)