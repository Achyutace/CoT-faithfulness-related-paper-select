# CoT Faithfulness Category

## 项目描述

先上传一下csv，项目仓库不定时更新。
本项目专注于对Chain of Thought (CoT) faithfulness相关论文的分类、分析和处理。CoT是一种让大型语言模型通过逐步推理来解决问题的技术，而faithfulness则指推理过程与最终答案的一致性和可靠性。该项目通过自动化脚本对论文进行多维度分类，包括领域、类型、指标等，帮助研究者快速筛选和分析相关文献。

## 文件夹结构

```
CoT_faithfulness_category/
├── code/                          # 代码文件夹
│   ├── step_11_sort_paper_parallel.py    # 并行排序论文脚本
│   ├── step_12_5pass4.py                 # 5-pass分类处理脚本
│   ├── step_2_category.py                # 论文分类脚本
│   └── tem_code/                         # 临时代码文件夹
│       ├── category.py                   # 分类相关代码
│       ├── generate_RDF.py               # RDF生成代码
│       ├── old_sort_paper.py             # 旧版排序代码
│       ├── relabel_survey.py             # 重新标注调查代码
│       ├── sort_paper.py                 # 排序论文代码
│       ├── tem.py                        # 临时代码
│       └── test_citation_range.py        # 测试引用范围代码
├── data/                          # 数据文件夹
│   ├── first_selected_papers.csv         # 初始选择的论文数据
│   ├── papers_filtered_smart.csv         # 智能过滤的论文数据
│   ├── papers_with_dimensions_v1.csv     # 带维度的论文数据版本1
│   ├── papers_with_dimensions_v2.csv     # 带维度的论文数据版本2
│   ├── papers_with_dimensions_v3.csv     # 带维度的论文数据版本3
│   ├── papers_with_dimensions.csv        # 带维度的论文数据
│   ├── backup/                           # 备份数据
│   │   ├── 1faithfulness_papers_full_survey copy_12.16 16_00.csv
│   │   ├── 2faithfulness_papers_full_survey 16 22.csv
│   │   ├── 3faithfulness_papers_full_survey 16 35.csv
│   │   ├── 4faithfulness_papers_full_survey 16 37.csv
│   │   ├── cot_faithfulness_citations_analysis.csv
│   │   ├── faithfulness_papers_full_survey.csv
│   │   ├── old_sort_by_category/         # 旧版按类别排序
│   │   └── old_sort_by_domain/           # 旧版按领域排序
│   ├── citations_select/                 # 引用选择文件夹
│   │   ├── cot_faithfulness_citations_4pass2.csv
│   │   ├── cot_faithfulness_citations_final_selection.csv
│   │   ├── cot_faithfulness_citations_select_11.csv
│   │   ├── cot_faithfulness_citations_select_12.csv
│   │   ├── cot_faithfulness_citations_select_13.csv
│   │   ├── cot_faithfulness_citations_select_14.csv
│   │   ├── cot_faithfulness_citations_select_15.csv
│   ├── sort_by_domain/                   # 按领域排序文件夹
│   │   ├── faithfulness_papers_code_v1.csv
│   │   ├── faithfulness_papers_code_v2.csv
│   │   ├── faithfulness_papers_code_v3.csv
│   │   ├── faithfulness_papers_general_v1.csv
│   │   ├── faithfulness_papers_general_v2.csv
│   │   ├── faithfulness_papers_general_v3.csv
│   │   ├── faithfulness_papers_logic_v1.csv
│   │   ├── faithfulness_papers_logic_v2.csv
│   │   ├── faithfulness_papers_logic_v3.csv
│   │   ├── faithfulness_papers_math_v1.csv
│   │   ├── faithfulness_papers_math_v2.csv
│   │   ├── faithfulness_papers_math_v3.csv
│   │   ├── faithfulness_papers_math.csv
│   │   ├── faithfulness_papers_medical_v1.csv
│   │   ├── faithfulness_papers_medical_v2.csv
│   │   ├── faithfulness_papers_medical_v3.csv
│   │   ├── faithfulness_papers_medical.csv
│   │   ├── faithfulness_papers_society_v1.csv
│   │   ├── faithfulness_papers_society_v2.csv
│   │   ├── faithfulness_papers_society_v3.csv
│   │   └── faithfulness_papers_society.csv
│   ├── sort_by_has_metrics/              # 按是否有指标排序
│   │   ├── faithfulness_papers_false_v1.csv
│   │   ├── faithfulness_papers_false_v2.csv
│   │   ├── faithfulness_papers_false_v3.csv
│   │   ├── faithfulness_papers_true_v1.csv
│   │   ├── faithfulness_papers_true_v2.csv
│   │   └── faithfulness_papers_true_v3.csv
│   ├── sort_by_has_phenomenon/           # 按是否有现象排序
│   ├── sort_by_mitigation_methods/       # 按缓解方法排序
│   ├── sort_by_related_score/            # 按相关分数排序
│   └── sort_by_type/                     # 按类型排序
├── citation_cache.json                   # 引用缓存文件
└── tem.md                               # 临时笔记文件
```

## 安装和依赖

### 环境要求
- Python 3.7+
- 推荐使用虚拟环境

### 安装依赖
看着安装，都是常用库

## 使用方法

暂

## 数据说明

- **论文数据**: 包含标题、作者、摘要、领域等信息
- **分类维度**: 
  - 领域 (domain): code, general, logic, math, medical, society
  - 类型 (type)
  - 是否有指标 (has_metrics)
  - 是否有现象 (has_phenomenon)
  - 缓解方法 (mitigation_methods)
  - 相关分数 (related_score)

## 贡献

欢迎提交Issue和Pull Request来改进本项目。项目仓库持续更新中。本readme由ai生成。

## 许可证

本项目采用MIT许可证。