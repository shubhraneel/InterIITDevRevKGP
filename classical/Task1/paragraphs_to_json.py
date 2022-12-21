import pandas as pd

df=pd.read_csv("data-dir/train_data.csv")

paragraphs = df['Paragraph']
para_dict = {}
doc_id = 0
for paragraph in paragraphs:
  if paragraph not in para_dict:
    para_dict[paragraph] = str(doc_id)
    doc_id += 1
  
paragraph_doc_id = list(para_dict.items())
paragraph_doc_id = pd.DataFrame(paragraph_doc_id, columns =['text', 'id'])
paragraph_doc_id.to_json("data-dir/paragraphs.json",orient="records",lines=True)

# df=df[['Paragraph']]
# df['id']=df.index.astype(str)

# df.rename({"Paragraph":"text"},axis=1,inplace=True)
# print(df.head())
# df.to_json("data-dir/paragraphs.json",orient="records",lines=True)

df['id'] = [para_dict[paragraph] for paragraph in paragraphs]
df=df[['Question', 'id']]
df.to_csv('data-dir/questions_only.csv')

