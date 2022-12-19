import pandas as pd
from gensim.models.doc2vec import Doc2Vec

def collate_paragraphs(df):
    collated_df=df.groupby('Theme',as_index=False).aggregate(lambda x: x.tolist())
    return collated_df

def co_appearance_on_collated_df(collated_df):
    co_appearance=[]
    for idx, row in collated_df.iterrows():
        topic_co_appearance=[]
        for para,ques in zip(row['Paragraph'],row['Question']):
            topic_co_appearance.append(len(set(para.split())&set(ques.split())))
        co_appearance.append(topic_co_appearance)
    collated_df['co_appearance_scores']=co_appearance
    return collated_df
    

if __name__ == "__main__":
    df=pd.read_csv("dataset/train_data.csv")
    collated_df=collate_paragraphs(df)
    collated_df=co_appearance_on_collated_df(collated_df)
    collated_df.to_csv("dataset/collated.csv")