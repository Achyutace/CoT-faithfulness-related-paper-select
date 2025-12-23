import csv
import json
import os
import re
import time
from typing import Optional

import requests


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "filtered_faithfulness_papers_v1.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "papers")


def sanitize_title_for_filename(title: str) -> str:
    """
    将论文标题转换成适合作为文件名的字符串：
    - 去掉首尾空格
    - 替换 Windows 不允许的字符：\/:*?"<>|
    - 压缩重复空格
    - 限制文件名长度，避免路径过长
    """
    title = title.strip()
    # 替换非法字符为下划线
    title = re.sub(r'[\\/:*?"<>|]', "_", title)
    # 压缩空白
    title = re.sub(r"\s+", " ", title)
    # 文件系统一般对单个文件名（不含路径）支持至少 200+ 字符，这里保守截断
    max_len = 150
    if len(title) > max_len:
        title = title[:max_len].rstrip()
    return title


def ensure_output_dir() -> None:
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_file(url: str, dest_path: str, timeout: int = 60) -> bool:
    """下载单个 PDF 文件。如果成功返回 True，否则 False。"""
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            # 简单判断内容类型
            content_type = r.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type and not dest_path.lower().endswith(".pdf"):
                # 如果服务器没有标明 pdf，但我们预期是 pdf，则仍然尝试保存
                pass

            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"  下载失败: {url} -> {dest_path}，原因: {e}")
        # 如有部分文件，删除
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        return False


def search_arxiv_pdf(title: str) -> Optional[str]:
    """
    使用 arXiv API 根据标题搜索论文，并返回 PDF 链接（若找到）。
    搜索策略：ti:"title" 精确标题搜索，取第一个结果。
    """
    base_url = "http://export.arxiv.org/api/query"
    query = f'ti:"{title}"'
    params = {
        "search_query": query,
        "start": 0,
        "max_results": 1,
    }
    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        print(f"  arXiv 搜索失败: {e}")
        return None

    # 粗糙解析：找到第一个 <id>http://arxiv.org/abs/xxxx</id>
    # 更严谨可以用 xml.etree.ElementTree，但这里保持依赖简单。
    m = re.search(r"<id>http://arxiv\.org/abs/([^<]+)</id>", text)
    if not m:
        return None
    arxiv_id = m.group(1)
    # 构造 pdf 链接
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return pdf_url


def search_semantic_scholar_pdf(title: str) -> Optional[str]:
    """
    使用 Semantic Scholar Graph API 搜索论文，并返回 open access PDF 链接（若有）。
    不需要 API key 的公开接口（有配额限制）。
    文档参考：https://api.semanticscholar.org/api-docs/graph
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "limit": 1,
        "fields": "title,openAccessPdf",
    }
    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Semantic Scholar 搜索失败: {e}")
        return None

    papers = data.get("data") or []
    if not papers:
        return None
    paper = papers[0]
    oa = paper.get("openAccessPdf") or {}
    url = oa.get("url")
    return url


def download_paper_for_title(title: str) -> None:
    """针对单个标题，尝试从 arXiv，然后 Semantic Scholar 下载 PDF。"""
    safe_name = sanitize_title_for_filename(title)
    dest_path = os.path.join(OUTPUT_DIR, f"{safe_name}.pdf")

    if os.path.exists(dest_path):
        print(f"[已存在] {title} -> {dest_path}")
        return

    print(f"[开始] {title}")

    # 1) 先试 arXiv
    pdf_url = search_arxiv_pdf(title)
    if pdf_url:
        print(f"  通过 arXiv 找到 PDF: {pdf_url}")
        if download_file(pdf_url, dest_path):
            print(f"[成功] arXiv 下载完成: {dest_path}")
            return

    # 2) 再试 Semantic Scholar
    pdf_url = search_semantic_scholar_pdf(title)
    if pdf_url:
        print(f"  通过 Semantic Scholar 找到 PDF: {pdf_url}")
        if download_file(pdf_url, dest_path):
            print(f"[成功] Semantic Scholar 下载完成: {dest_path}")
            return

    print(f"[失败] 未能为该标题找到可下载的 PDF: {title}")


def load_titles_from_csv(csv_path: str) -> list[str]:
    titles: list[str] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        print(reader.fieldnames)  
        if "title" not in reader.fieldnames:
            raise ValueError("CSV 文件中未找到 'title' 列")
        for row in reader:
            t = (row.get("title") or "").strip()
            if t:
                titles.append(t)
    return titles


def main(sleep_between: float = 2.0) -> None:
    """
    主流程：
    1. 读取 CSV 中的所有 title
    2. 对每个 title 依次尝试下载 PDF 到 data/papers/title.pdf
    """
    print(f"CSV 路径: {CSV_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    ensure_output_dir()

    titles = load_titles_from_csv(CSV_PATH)
    print(f"共读取到 {len(titles)} 篇论文标题。")

    for idx, title in enumerate(titles, start=1):
        print(f"\n=== [{idx}/{len(titles)}] {title} ===")
        download_paper_for_title(title)
        # 简单限速，避免触发 API 限制
        if sleep_between > 0:
            time.sleep(sleep_between)


if __name__ == "__main__":
    main()


