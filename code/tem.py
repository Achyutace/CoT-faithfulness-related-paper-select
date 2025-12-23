import ast
import csv
import os
from typing import Any, Dict, List

from xml.sax.saxutils import escape


# 基本路径设置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUP_CSV = os.path.join(BASE_DIR, "data", "backup", "cot_faithfulness_citations_analysis.csv")
FILTER_CSV = os.path.join(BASE_DIR, "data", "filtered_faithfulness_papers_v1.csv")
OUTPUT_RDF = os.path.join(BASE_DIR, "data", "cot_faithfulness_filtered.rdf")


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


def main() -> None:
    print(f"读取过滤列表: {FILTER_CSV}")
    print(f"读取备份数据: {BACKUP_CSV}")
    print(f"输出 RDF: {OUTPUT_RDF}")

    filter_titles = set(read_titles(FILTER_CSV))
    print(f"过滤列表标题数: {len(filter_titles)}")

    backup_rows = read_backup_rows(BACKUP_CSV)
    matched_rows = [row for row in backup_rows if (row.get("title") or "").strip() in filter_titles]

    print(f"备份总记录: {len(backup_rows)}，匹配到过滤列表的记录: {len(matched_rows)}")

    generate_rdf(matched_rows, OUTPUT_RDF)
    print(f"已生成 RDF: {OUTPUT_RDF} （仅包含过滤列表中的论文，可在 Zotero 中尝试导入）")


if __name__ == "__main__":
    main()


