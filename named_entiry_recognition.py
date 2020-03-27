# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:48:10 2019

@author: smouz

"""
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nltk
from nltk import word_tokenize, sent_tokenize



print('Current working dir:', os.getcwd())


#%%
#               Named Entity Recognition (NER) with NLTK
# =============================================================================

article = ''
article_path = os.path.join('News articles', 'articles.txt')
with open(article_path,'r', encoding='utf-8') as txt:
    article = txt.read()


# Tokenize the article into sentences: sentences
sentences = sent_tokenize(article)

# Tokenize each sentence into words: token_sentences
token_sentences = [word_tokenize(sent) for sent in sentences]

# Tag each tokenized sentence into parts of speech: pos_sentences
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences]

# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)

# Test for stems of the tree with 'NE' tags
all_labels = []
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label"):
            all_labels.append(chunk)
        if hasattr(chunk, "label") and chunk.label() == "NN":
            print(chunk)

all_labels[20]
all_labels[0].height()

#%%
#                                Charting
# =============================================================================
# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)

# Create the nested for loop
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] += 1

# Create a list from the dictionary keys for the chart labels: labels
labels = list(ner_categories.keys())

# Create a list of the values: values
values = [ner_categories.get(l) for l in labels]

# Create the pie chart
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

# Display the chart
plt.show()



#%%
#                                   SpaCy NLP
# =============================================================================

# Import spacy
import spacy

# To minimize execution times, you'll be asked to specify the keyword arguments
# Instantiate the English model: nlp
nlp = spacy.load('en', tagger=False, parser=False, matcher=False)

# Create a new document: doc
doc = nlp(article)

# Print all of the found entities and their labels
for ent in doc.ents:
    print(ent.label_, ent.text)



#%%
#                                 PolyGlot
# =============================================================================
"""LINUX ONLY"""

#                          French NER with polyglot
# =============================================================================
from polyglot.text import Text

# Create a new text object using Polyglot's Text class: txt
txt = Text(article)

# Print each of the entities found
for ent in txt.entities:
    print(ent)

# Print the type of ent
print(type(ent))


#%%
#
# =============================================================================
# Create the list of tuples: entities
entities = [(ent.tag, ' '.join(ent)) for ent in txt.entities]

# Print entities
print(entities)


#%%
# Spanish NER with polyglot
# =============================================================================
# Initialize the count variable: count
count = 0

# Iterate over all the entities
for ent in txt.entities:
    # Check whether the entity contains 'Márquez' or 'Gabo'
    if "Márquez" in ent or "Gabo" in ent:
        # Increment count
        count += 1

# Print count
print(count)

# Calculate the percentage of entities that refer to "Gabo": percentage
percentage = count / len(txt.entities)
print(percentage)



