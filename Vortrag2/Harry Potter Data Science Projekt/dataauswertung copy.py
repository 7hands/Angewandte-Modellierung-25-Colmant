import pandas as pd
import numpy as np
import spacy
import os
import re


nlp = spacy.load("en_core_web_sm") # English Language model
nlp.max_length = 1800000

all_books = [b for b in os.scandir('data')if '.txt' in b.name]

book = all_books[0]
print(all_books[0])
book_text = open(book, encoding="utf8").read()
book_doc = nlp(book_text)



#html = displacy.render(book_doc[0:2000], style="ent")

#with open ("data.html", "w") as f:
#    f.write(html)

#dataset

chars_df = pd.read_csv("extra_chars.csv")

chars_df['character'] = chars_df['character'].apply(lambda x: re.sub("[\(].*?[\)]", "", x)) 
chars_df['character_firstname'] = chars_df['character'].apply(lambda x: x.split(' ', 1)[0])


#get named entity list per sentence
sent_entity_df = []

for sent in book_doc.sents:
    entity_list = [ent.text for ent in sent.ents]
    sent_entity_df.append({"sentence": sent, "entities": entity_list})

sent_entity_df = pd.DataFrame(sent_entity_df)

def filter_entity(ent_list, chars_df):
    return [ent for ent in ent_list 
            if ent in list(chars_df.character) 
            ]

sent_entity_df['character_entities'] = sent_entity_df['entities'].apply(lambda x: filter_entity(x, chars_df))

sent_entity_df_filtered = sent_entity_df[sent_entity_df['character_entities'].map(len) > 0]


sent_entity_df_filtered.to_csv('data_chars2.csv', index=False)

#create relationships
window_size = 5
relationships = []

for i in range(sent_entity_df_filtered.index[-1]):
    end_i = min(i+5, sent_entity_df_filtered.index[-1])
    char_list = sum((sent_entity_df_filtered.loc[i:end_i].character_entities), [])

    char_unique = [char_list[i] for i in range(len(char_list))
                    if (i==0) or char_list[i] != char_list[i-1] ]

    if len(char_unique) > 1:
        for idx, a in enumerate(char_unique[:-1]):
            b= char_unique[idx +1]
            relationships.append({"source": a, "target": b})
relationship_df = pd.DataFrame(relationships)

relationship_df = pd.DataFrame(np.sort(relationship_df.values, axis = 1), columns = relationship_df.columns)
relationship_df["value"] = 1
relationship_df = relationship_df.groupby(["source", "target"], sort=False, as_index=False).sum()


relationship_df.to_csv('relationships_b1_ex.csv', index=False)


#visualization

