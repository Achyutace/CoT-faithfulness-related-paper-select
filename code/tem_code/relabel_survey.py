import pandas as pd

df = pd.read_csv('faithfulness_papers_full_survey.csv')
# 如果标题中有“Survey”或“Review”，则标记为 Survey 类别
df['Category'] = df.apply(lambda row: 'Survey' if 'survey' in row['Title'].lower() or 'review' in row['Title'].lower() else row['Category'], axis=1)
# 筛选出benchmark
df['Category'] = df.apply(lambda row: 'Benchmark' if 'benchmark' in row['Title'].lower() else row['Category'], axis=1)


# 如果标题有medical相关词汇，则标记为 Medical 领域
def domain_is_medical(title):
    medical_keywords = ['medical', 'health', 'clinical', 'disease', 'patient', 'diagnosis', 'treatment']
    title_lower = title.lower()
    return any(keyword in title_lower for keyword in medical_keywords)
# 如果df['Domain']是general且标题有医学，则改为medical
df['Domain'] = df.apply(lambda row: 'Medical' if row['Domain'] == 'General' and domain_is_medical(row['Title']) else row['Domain'], axis=1)

df.to_csv('new_faithfulness_papers_full_survey.csv', index=False)