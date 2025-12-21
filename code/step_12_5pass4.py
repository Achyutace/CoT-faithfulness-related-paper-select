'''
code.5pas43 的 Docstring
从初筛中选出最终的文件：cot_faithfulness_citations_final_selection.csv

'''


import pandas as pd
import os

DATA_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/data/citations_select"

df1 = pd.read_csv(DATA_PATH + "/cot_faithfulness_citations_select_11.csv")
df2 = pd.read_csv(DATA_PATH + "/cot_faithfulness_citations_select_12.csv")
df3 = pd.read_csv(DATA_PATH + "/cot_faithfulness_citations_select_13.csv")
df4 = pd.read_csv(DATA_PATH + "/cot_faithfulness_citations_select_14.csv")
df5 = pd.read_csv(DATA_PATH + "/cot_faithfulness_citations_select_15.csv")
#看一下df2、df3、df4、df5中分别有几个is_relevant为true但is_CoT_relavent或者is_faithfulness_realavent为false行
wtf_list_2 = df2[(df2['is_relevant'] == True) & ((df2['is_CoT_relevant'] == False) | (df2['is_faithfulness_relevant'] == False))]
wtf_list_3 = df3[(df3['is_relevant'] == True) & ((df3['is_CoT_relevant'] == False) | (df3['is_faithfulness_relevant'] == False))]
wtf_list_4 = df4[(df4['is_relevant'] == True) & ((df4['is_CoT_relevant'] == False) | (df4['is_faithfulness_relevant'] == False))]
wtf_list_5 = df5[(df5['is_relevant'] == True) & ((df5['is_CoT_relevant'] == False) | (df5['is_faithfulness_relevant'] == False))]
# print("df2 wtf amount:", len(wtf_list_2))
# print("df3 wtf amount:", len(wtf_list_3))
# print("df4 wtf amount:", len(wtf_list_4))
# print("df5 wtf amount:", len(wtf_list_5))
# 打印后发现为0，证明这个方法还是蛮鲁棒的。

# 选出所有is_relavent有至少两个true的标题，并统计个数
combined_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
true_counts = combined_df[combined_df['is_relevant'] == True]['title'].value_counts()
filtered_titles = true_counts[true_counts >= 4].index.tolist()
print("filtered_amount:", len(filtered_titles))
# 筛选出filtered_titles对应的行，按照12，13，14这三个文件（他们的csv头是一样的）。然后手动处理一下列。
# has_phenomenon、has_metrics、mitigation_methods按照3pass2保留
# 先提取12的表头
result_rows = []

for idx, title in enumerate(filtered_titles):
    row_12 = df2[df2['title'] == title].iloc[0]
    row_13 = df3[df3['title'] == title].iloc[0]
    row_14 = df4[df4['title'] == title].iloc[0]
    row_15 = df5[df5['title'] == title].iloc[0]
    # 按照原来row_合并成新的行
    new_row = row_12.copy()
    if idx == 1:
        print(row_12["has_phenomenon"])
    has_phenomenon = (bool(row_12['has_phenomenon']) + bool(row_13['has_phenomenon']) + bool(row_14['has_phenomenon']) + bool(row_15['has_phenomenon'])) >= 4
    has_metrics = (bool(row_12['has_metrics']) + bool(row_13['has_metrics']) + bool(row_14['has_metrics']) + bool(row_15['has_metrics'])) >= 4
    # print("has_phenomenon: ", has_phenomenon, "has_metrics:", has_metrics)
    new_row['has_phenomenon'] = has_phenomenon
    new_row['has_metrics'] = has_metrics
    # 把"Verification & External Tools, Prompting & In-Context Learning"，"Training & Fine-tuning, Verification & External Tools"这种提取成字典，值为True/False
    # key只有可能是：Training & Fine-tuning，Verification & External Tools，Prompting & In-Context Learning，Interpretability & Internal Mechanisms，Consistency & Ensembling这五个
    # 这个采用3pass1的方法，只要有一个True就算True
    mitigation_methods_dict = {
        "Training & Fine-tuning": False,
        "Verification & External Tools": False,
        "Prompting & In-Context Learning": False,
        "Interpretability & Internal Mechanisms": False,
        "Consistency & Ensembling": False
    }

    for method_str in [row_12['mitigation_methods'], row_13['mitigation_methods'], row_14['mitigation_methods'], row_15['mitigation_methods']]:
        
        if method_str != method_str:  # 检查是否为NaN
            continue
        methods = [m.strip() for m in method_str.split(",")]
        for method in methods:
            if method in mitigation_methods_dict:
                mitigation_methods_dict[method] = True
    
    
    # 把mitigation_methods_dict转成字符串存储
    mitigation_methods_list = [k for k, v in mitigation_methods_dict.items() if v]
    new_row['mitigation_methods'] = ", ".join(mitigation_methods_list)
    result_rows.append(new_row)

result_df = pd.DataFrame(result_rows)
# 删去 is_CoT_relevant、is_faithfulness_relevant 列
result_df = result_df.drop(columns=['is_CoT_relevant', 'is_faithfulness_relevant'])
# 保存结果
result_df.to_csv(DATA_PATH + "/cot_faithfulness_citations_final_selection.csv", index=False)

    




