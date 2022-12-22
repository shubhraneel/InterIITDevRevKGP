import pandas as pd
import os
# use numexpr to speed up df.query

df_ = pd.read_csv("data-dir/train_data.csv")

themes=df_['Theme'].unique()

for theme in themes:
	df= df_.query(f"Theme ==  '{theme}'")
	print(df)
	paragraphs = df['Paragraph']
	para_dict = {}
	doc_id = 0
	for paragraph in paragraphs:
		if paragraph not in para_dict:
			para_dict[paragraph] = str(doc_id)
			doc_id += 1

	paragraph_doc_id = list(para_dict.items())
	paragraph_doc_id = pd.DataFrame(paragraph_doc_id, columns=['text', 'id'])
	os.mkdir(f"data-dir/theme_wise/{theme.casefold()}")
	paragraph_doc_id.to_json(f"data-dir/theme_wise/{theme.casefold()}/paragraphs.json",
							orient="records", lines=True)
	df['id'] = [para_dict[paragraph] for paragraph in paragraphs]
	df = df[['Question', 'id']]
	df['Theme']=[theme]*df.shape[0]
	df.to_csv(f'data-dir/theme_wise/{theme.casefold()}/questions_only.csv')
