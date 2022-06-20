#!/usr/bin/env python
# coding: utf-8

# In[4]:


import spacy
import pandas as pd 
# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

base_path = '/home/manju/Desktop/assign/task_data/'

def readInputFile(csv_file):
    data = pd.read_csv(csv_file)
    transcript = data['transcription']
    return transcript

def main(base_path):
    train_file = base_path + 'train_data.csv'
    valid_file = base_path + 'valid_data.csv'

    transcript = readInputFile(valid_file)
    print("Transcription analysis: Noun Phrases and Verbs are printed, if there are any NER present in the sentense they will be printed ")

    # Process documents one by one
    for text in transcript:
        doc = nlp(text) 
    
        print("\nSentence: \"", text, end = '\"; ')
        # Analyze syntax
        print("Nouns:", [chunk.text for chunk in doc.noun_chunks], end = '; ')
        print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"], end = '; ')

        # Find named entities, phrases and concepts    
        for entity in doc.ents:
            print("NER:",entity.text, entity.label_, end = '; ')   
            
if __name__ == "__main__":
    main(base_path)


# In[ ]:




