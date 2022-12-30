import pandas as pd
import os
import json 
# use numexpr to speed up df.query

# Theme wise:
# df_ = pd.read_csv("data-dir/train_data.csv")
# themes = df_['Theme'].unique()

# for theme in themes:
#     df = df_.query(f"Theme ==  '{theme}'")
#     print(df)
#     paragraphs = df['Paragraph']
#     para_dict = {}
#     doc_id = 0
#     for paragraph in paragraphs:
#         if paragraph not in para_dict:
#             para_dict[paragraph] = str(doc_id)
#             doc_id += 1

#     paragraph_doc_id = list(para_dict.items())
#     paragraph_doc_id = pd.DataFrame(paragraph_doc_id, columns=['text', 'id'])
#     os.mkdir(f"data-dir/theme_wise/{theme.casefold()}")
#     paragraph_doc_id.to_json(f"data-dir/theme_wise/{theme.casefold()}/paragraphs.json",
#                              orient="records", lines=True)
#     df['id'] = [para_dict[paragraph] for paragraph in paragraphs]
#     df = df[['Question', 'id']]
#     df['Theme'] = [theme]*df.shape[0]
#     df.to_csv(f'data-dir/theme_wise/{theme.casefold()}/questions_only.csv')

# All at once
df = pd.read_csv("data-dir/train_data.csv")
df = df.drop_duplicates(subset=["Unnamed: 0"]).drop_duplicates(
    subset=["Question"]).reset_index(drop=True)

themes = df['Theme'].unique()
theme_dict = {}
theme_id = 0
for theme in themes:
    if theme not in theme_dict:
        theme_dict[theme] = str(theme_id)
        theme_id += 1

paragraphs = df['Paragraph']
para_dict = {}
doc_id = 0
para_theme_dict={}
for idx,row in df.iterrows():
    if row['Paragraph'] not in para_dict:
        para_dict[row['Paragraph']] = str(doc_id)
        para_theme_dict[para_dict[row['Paragraph']]]=theme_dict[row['Theme']]
        doc_id += 1

with open("data-dir/para_theme.json","w") as of:
    json.dump(para_theme_dict,of)   


paragraph_doc_id = list(para_dict.items())
paragraph_doc_id = pd.DataFrame(paragraph_doc_id, columns=['text', 'id'])
paragraph_doc_id.to_json(f"data-dir/paragraphs.json",
                         orient="records", lines=True)

df['id'] = [para_dict[paragraph] for paragraph in paragraphs]
df['theme_id'] = [theme_dict[theme] for theme in df['Theme']]

df1 = df[['Question', 'id', 'Answer_possible', 'theme_id']]
df1.to_csv(f'data-dir/questions_only.csv', index=False)
df['Question'] = df['Theme']+' '+df['Question']
df = df[['Question', 'id', 'Answer_possible', 'theme_id']]
df.to_csv(f'data-dir/questions_theme_concat.csv', index=False)

