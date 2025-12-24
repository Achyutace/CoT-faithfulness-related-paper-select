import ast
import csv
import os
import re
import json
from typing import Any, Dict, List

from xml.sax.saxutils import escape


# 基本路径设置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUP_CSV = os.path.join(BASE_DIR, "data", "backup", "cot_faithfulness_citations_analysis.csv")
FILTER_CSV = os.path.join(BASE_DIR, "data", "filtered_faithfulness_papers_v1.csv")
PAPERS_DIR = os.path.join(BASE_DIR, "data", "papers")
OUTPUT_RDF = os.path.join(BASE_DIR, "data", "cot_faithfulness_with_pdf.rdf")
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "cot_faithfulness_with_pdf.json")
OUTPUT_RIS = os.path.join(BASE_DIR, "data", "cot_faithfulness_with_pdf.ris")


def sanitize_title_for_filename(title: str) -> str:
    """与下载脚本一致的文件名清洗。"""
    title = title.strip()
    title = re.sub(r'[\\/:*?"<>|]', "_", title)
    title = re.sub(r"\s+", " ", title)
    if len(title) > 150:
        title = title[:150].rstrip()
    return title


def read_titles(csv_path: str) -> List[str]:
    """读取 filtered_faithfulness_papers_v1.csv 的 title 列，返回去重列表（处理 BOM）。"""
    titles: List[str] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "title" not in (reader.fieldnames or []):
            raise ValueError(f"CSV 缺少 'title' 列，实际列名: {reader.fieldnames}")
        for row in reader:
            t = (row.get("title") or "").strip()
            if t:
                titles.append(t)
    return list(dict.fromkeys(titles))  # 去重并保持顺序


def read_backup_rows(csv_path: str) -> List[Dict[str, Any]]:
    """
    读取备份 CSV（包含 title, year, abstract, publication_venue 等列）。
    使用 utf-8-sig 处理可能的 BOM。
    """
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = ["title", "year", "abstract", "publication_venue"]
        for col in required:
            if col not in (reader.fieldnames or []):
                raise ValueError(f"CSV 缺少必需列: {col}, 实际列: {reader.fieldnames}")
        for row in reader:
            rows.append(row)
    return rows


def parse_publication_venue(raw: str) -> Dict[str, Any]:
    """
    publication_venue 字段是字符串形式的 dict，需谨慎解析。
    使用 ast.literal_eval 以避免执行任意代码。
    返回包含 name / id / type / url 等字段。
    """
    venue: Dict[str, Any] = {"name": "", "id": "", "type": "", "url": ""}
    if not raw:
        return venue
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            venue["name"] = parsed.get("name") or ""
            venue["id"] = parsed.get("id") or ""
            venue["type"] = parsed.get("type") or ""
            venue["url"] = parsed.get("url") or ""
    except Exception:
        # 解析失败时保留空字符串，避免中断
        pass
    return venue


def generate_rdf(rows: List[Dict[str, Any]], output_path: str) -> None:
    """
    将筛选后的数据生成 RDF，适合尝试导入 Zotero。
    使用 Dublin Core 基本字段。
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        )

        for idx, row in enumerate(rows, start=1):
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            year = (row.get("year") or "").strip()
            venue_raw = row.get("publication_venue") or ""
            venue = parse_publication_venue(venue_raw)

            # 构造唯一 about，避免相同 URL 被合并丢失
            about = f"urn:paper:{idx}"

            f.write(f'  <rdf:Description rdf:about="{escape(about)}">\n')
            if title:
                f.write(f"    <dc:title>{escape(title)}</dc:title>\n")
            if abstract:
                f.write(f"    <dc:description>{escape(abstract)}</dc:description>\n")
            if year:
                f.write(f"    <dc:date>{escape(year)}</dc:date>\n")
            if venue.get("name"):
                f.write(f"    <dc:source>{escape(venue['name'])}</dc:source>\n")
            # 额外存放 venue url/id 作为 identifier，防止被当作 about 去重
            if venue.get("url"):
                f.write(f"    <dc:identifier>{escape(venue['url'])}</dc:identifier>\n")
            elif venue.get("id"):
                f.write(f"    <dc:identifier>{escape(venue['id'])}</dc:identifier>\n")
            f.write("  </rdf:Description>\n")

        f.write("</rdf:RDF>\n")


def generate_json_with_attachments(rows: List[Dict[str, Any]], output_path: str) -> None:
    """
    生成 Zotero 可导入的 JSON，每条记录附加本地 PDF 的绝对路径。
    """
    items = []
    for row in rows:
        title = (row.get("title") or "").strip()
        abstract = (row.get("abstract") or "").strip()
        year = (row.get("year") or "").strip()
        venue_raw = row.get("publication_venue") or ""
        venue = parse_publication_venue(venue_raw)

        fname = sanitize_title_for_filename(title) + ".pdf"
        fpath = os.path.join(PAPERS_DIR, fname)
        abs_path = os.path.abspath(fpath)

        item = {
            "itemType": "journalArticle",
            "title": title,
            "abstractNote": abstract,
            "date": year,
            "publicationTitle": venue.get("name") or "",
            "attachments": [
                {
                    "path": abs_path,
                    "title": "PDF",
                    "mimeType": "application/pdf",
                    # Zotero 会把 path 作为“链接的文件附件”
                }
            ],
        }
        items.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def generate_ris_with_attachments(rows: List[Dict[str, Any]], output_path: str) -> None:
    """
    生成 RIS（Zotero 支持），用 L1 写入本地 PDF 绝对路径。
    仅包含本地已有 PDF 的记录。
    """
    lines: List[str] = []
    for row in rows:
        title = (row.get("title") or "").strip()
        abstract = (row.get("abstract") or "").strip()
        year = (row.get("year") or "").strip()
        venue_raw = row.get("publication_venue") or ""
        venue = parse_publication_venue(venue_raw)
        journal = venue.get("name") or ""

        fname = sanitize_title_for_filename(title) + ".pdf"
        fpath = os.path.join(PAPERS_DIR, fname)
        abs_path = os.path.abspath(fpath)

        lines.append("TY  - JOUR")
        if title:
            lines.append(f"TI  - {title}")
        if journal:
            lines.append(f"JO  - {journal}")
        if year:
            lines.append(f"PY  - {year}")
        if abstract:
            lines.append(f"AB  - {abstract}")
        # L1: file attachment (path)
        lines.append(f"L1  - {abs_path}")
        lines.append("ER  - ")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    print(f"读取过滤列表: {FILTER_CSV}")
    print(f"读取备份数据: {BACKUP_CSV}")
    print(f"PDF 目录: {PAPERS_DIR}")
    print(f"输出 RDF: {OUTPUT_RDF}")
    print(f"输出 JSON: {OUTPUT_JSON}")
    print(f"输出 RIS: {OUTPUT_RIS}")

    filter_titles = set(read_titles(FILTER_CSV))
    print(f"过滤列表标题数: {len(filter_titles)}")

    backup_rows = read_backup_rows(BACKUP_CSV)
    matched_rows = [row for row in backup_rows if (row.get("title") or "").strip() in filter_titles]
    print(f"备份总记录: {len(backup_rows)}，匹配到过滤列表的记录: {len(matched_rows)}")

    available_rows: List[Dict[str, Any]] = []
    for row in matched_rows:
        title = (row.get("title") or "").strip()
        fname = sanitize_title_for_filename(title) + ".pdf"
        fpath = os.path.join(PAPERS_DIR, fname)
        if os.path.exists(fpath):
            available_rows.append(row)

    print(f"匹配且有 PDF 的记录: {len(available_rows)}")

    generate_rdf(available_rows, OUTPUT_RDF)
    print(f"已生成 RDF: {OUTPUT_RDF} （仅包含本地已有 PDF 的论文，可在 Zotero 中尝试导入）")

    generate_json_with_attachments(available_rows, OUTPUT_JSON)
    print(f"已生成 JSON: {OUTPUT_JSON} （可在 Zotero 导入，自动挂载本地 PDF）")

    generate_ris_with_attachments(available_rows, OUTPUT_RIS)
    print(f"已生成 RIS: {OUTPUT_RIS} （Zotero 导入 RIS，附件通过 L1 指向本地 PDF）")


if __name__ == "__main__":
    main()


