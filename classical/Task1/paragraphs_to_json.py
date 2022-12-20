import pandas as pd

df=pd.read_csv("data-dir/train_data.csv")
df=df[['Paragraph']]
df['id']=df.index.astype(str)

df.rename({"Paragraph":"text"},axis=1,inplace=True)
print(df.head())
df.to_json("data-dir/paragraphs.json",orient="records",lines=True)

df=pd.read_csv("data-dir/train_data.csv")
df=df[['Question']]
df['id']=df.index
df.to_csv('data-dir/questions_only.csv')

