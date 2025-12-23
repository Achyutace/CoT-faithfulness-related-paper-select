import os
import re
import csv
import time
from typing import Optional

import requests


# 路径配置（基于当前仓库结构）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILTER_CSV = os.path.join(BASE_DIR, "data", "filtered_faithfulness_papers_v1.csv")
PAPERS_DIR = os.path.join(BASE_DIR, "data", "papers")


def sanitize_title_for_filename(title: str) -> str:
    """
    将论文标题转换为下载时使用的文件名（与下载脚本保持一致）：
    - 去掉首尾空格
    - 替换 Windows 不允许的字符：\/:*?"<>|
    - 压缩重复空格
    - 限制长度以避免过长路径
    """
    title = title.strip()
    title = re.sub(r'[\\/:*?"<>|]', "_", title)
    title = re.sub(r"\s+", " ", title)
    max_len = 150
    if len(title) > max_len:
        title = title[:max_len].rstrip()
    return title


def load_titles(csv_path: str) -> list[str]:
    """读取 filtered_faithfulness_papers_v1.csv 的 title 列（处理 BOM）。"""
    titles: list[str] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "title" not in (reader.fieldnames or []):
            raise ValueError(f"CSV 缺少 'title' 列，实际列名: {reader.fieldnames}")
        for row in reader:
            t = (row.get("title") or "").strip()
            if t:
                titles.append(t)
    return titles


def download_file(url: str, dest_path: str, timeout: int = 60) -> bool:
    """下载单个 PDF 文件。如果成功返回 True，否则 False。"""
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type and not dest_path.lower().endswith(".pdf"):
                # 即便未标明 pdf，也尝试保存
                pass
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"  下载失败: {url} -> {dest_path}，原因: {e}")
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        return False


def search_semantic_scholar_pdf(title: str, max_retries: int = 3, base_delay: float = 3.0) -> Optional[str]:
    """
    使用 Semantic Scholar Graph API 搜索论文，并返回 open access PDF 链接（若有）。
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "limit": 1,
        "fields": "title,openAccessPdf",
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(base_url, params=params, timeout=30)
            if resp.status_code == 429:
                delay = base_delay * attempt
                print(f"  命中 429，等待 {delay:.1f}s 后重试 ({attempt}/{max_retries})")
                time.sleep(delay)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt >= max_retries:
                print(f"  Semantic Scholar 请求失败（已重试 {max_retries} 次）：{e}")
                return None
            delay = base_delay * attempt
            print(f"  请求异常，等待 {delay:.1f}s 后重试 ({attempt}/{max_retries})：{e}")
            time.sleep(delay)

    papers = data.get("data") or []
    if not papers:
        return None
    paper = papers[0]
    oa = paper.get("openAccessPdf") or {}
    return oa.get("url")


def search_arxiv_pdf(title: str, max_retries: int = 3, base_delay: float = 3.0) -> Optional[str]:
    """
    使用 arXiv API 根据标题搜索论文，并返回 PDF 链接（若找到）。
    搜索策略：ti:"title" 精确标题搜索，取第一个结果。
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f'ti:"{title}"',
        "start": 0,
        "max_results": 1,
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(base_url, params=params, timeout=30)
            if resp.status_code == 429:
                delay = base_delay * attempt
                print(f"  arXiv 命中 429，等待 {delay:.1f}s 后重试 ({attempt}/{max_retries})")
                time.sleep(delay)
                continue
            resp.raise_for_status()
            text = resp.text
            break
        except Exception as e:
            if attempt >= max_retries:
                print(f"  arXiv 请求失败（已重试 {max_retries} 次）：{e}")
                return None
            delay = base_delay * attempt
            print(f"  arXiv 请求异常，等待 {delay:.1f}s 后重试 ({attempt}/{max_retries})：{e}")
            time.sleep(delay)

    # 粗糙解析第一个 id
    m = re.search(r"<id>http://arxiv\.org/abs/([^<]+)</id>", text)
    if not m:
        return None
    arxiv_id = m.group(1)
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def main() -> None:
    print(f"检查 CSV: {FILTER_CSV}")
    print(f"检查目录: {PAPERS_DIR}")

    titles = load_titles(FILTER_CSV)
    missing: list[str] = []

    for title in titles:
        fname = sanitize_title_for_filename(title) + ".pdf"
        fpath = os.path.join(PAPERS_DIR, fname)
        if not os.path.exists(fpath):
            missing.append(title)

    print(f"总计 {len(titles)} 篇，未找到 {len(missing)} 篇。")
    if missing:
        print("\n未找到的标题：")
        for t in missing:
            print(f"- {t}")

    # 尝试补充下载缺失的
    if missing:
        print("\n开始尝试从 Semantic Scholar 补充下载缺失的 PDF ...")
        os.makedirs(PAPERS_DIR, exist_ok=True)
        for idx, title in enumerate(missing, start=1):
            safe_name = sanitize_title_for_filename(title)
            dest_path = os.path.join(PAPERS_DIR, f"{safe_name}.pdf")
            print(f"\n[{idx}/{len(missing)}] {title}")
            pdf_url = None
            source = "Semantic Scholar"
            if not pdf_url:
                pdf_url = search_arxiv_pdf(title)
                source = "arXiv"
            if not pdf_url:
                print("  未找到可下载的 PDF 链接。")
                continue
            print(f"  发现 PDF 链接（{source}）：{pdf_url}")
            if download_file(pdf_url, dest_path):
                print(f"  下载成功：{dest_path}")
            else:
                print("  下载失败。")


if __name__ == "__main__":
    main()

