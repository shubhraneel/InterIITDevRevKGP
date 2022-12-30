import pandas as pd
import os
import json 

def reformat_data_for_sqlite(df, split):
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

    with open("data-dir/{}_para_theme.json".format(split),"w") as of:
        json.dump(para_theme_dict,of)   

    paragraph_doc_id = list(para_dict.items())
    paragraph_doc_id = pd.DataFrame(paragraph_doc_id, columns=['text', 'id'])
    paragraph_doc_id.to_json("data-dir/{}_paragraphs.json".format(split),
                            orient="records", lines=True)

    df['id'] = [para_dict[paragraph] for paragraph in paragraphs]
    df['theme_id'] = [theme_dict[theme] for theme in df['Theme']]

    df1 = df[['Question', 'id', 'Answer_possible', 'theme_id']]
    df1.to_csv('data-dir/{}_questions_only.csv'.format(split), index=False)
    df['Question'] = df['Theme']+' '+df['Question']
    df = df[['Question', 'id', 'Answer_possible', 'theme_id']]
    df.to_csv('data-dir/{}_questions_theme_concat.csv'.format(split), index=False)
