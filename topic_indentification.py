# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:00:04 2019

@author: smouz

"""

#%%
import os
import re

import numpy as np
import pandas as pd

from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize


print('Current working dir:', os.getcwd())

#%%
article = ''
with open('article.txt', 'r') as txt:
    article = txt.read()

#%%
#                       Basic Topic Identification
# =============================================================================

# Import Counter
from collections import Counter

# Tokenize the article: tokens
tokens = word_tokenize(article)

# Convert the tokens into lowercase: lower_tokens
lower_tokens = [t.lower() for t in tokens]

# Create a Counter with the lowercase tokens: bow_simple
bow_simple = Counter(lower_tokens)

# Print the 10 most common tokens
print(bow_simple.most_common(10))

#%%
# Remove some of the more common words
pop_words = ['the', 'in', ',', '.', 'a', 'an', 'of', 'and', 'is', "''", 'to']
for t in pop_words:
    bow_simple.pop(t)

print(bow_simple.most_common(10))



#%%
#                           Text preprocessing
# =============================================================================
# Import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

english_stops = stopwords.words('english')

# Retain alphabetic words: alpha_only
# removes punctuations
alpha_only = [t for t in lower_tokens if t.isalpha()]

# Remove all stop words: no_stops
no_stops = [t for t in alpha_only if t not in english_stops]

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

# Create the bag-of-words: bow
bow = Counter(lemmatized)

# Print the 10 most common tokens
print(bow.most_common(10))

#%%

def process_text(text_doc):
    """
    Convert to lowercase.
    Retain only alphabetical characters.
    Remove stop words.
    Lemmatize tokens into a new list.

    Returns:
    --------
        Lemmatized list
    """
    lower_tokens = [t.lower() for t in word_tokenize(text_doc)]
    alpha_only = [t for t in lower_tokens if t.isalpha()]
    no_stops = [t for t in alpha_only if t not in english_stops]
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
    return lemmatized

#%%
# open text files in directory for processing
def combine_articles():
    os.chdir('Wikipedia articles')
    articles = []
    for item in os.listdir():
        with open(item, 'r', encoding='utf-8') as txt:
            articles.append(txt.read())
    os.chdir('..')
    return articles
#%%
def tokenize(text):
    # load stopwords
    stop_words = stopwords.words("english")

    # remove punctuations (retain alphabetical and numeric chars) and convert to all lower case
    # tokenize resulting text
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", ' ', text.lower().strip()))

    # lemmatize and remove stop words
    lemmatized = [PorterStemmer().stem(word) for word in tokens if word not in stop_words]

    return lemmatized
#%%
articles = combine_articles()
# process each file w/for-loop
processed_files = list(map(process_text, articles))



#%%
sent_2 = articles[3][:200]
sent = "I am listening to Radio Dabon. We need help and doctors in that area. \
We are unable to move around because we have no gas. Dabon is located close \
to Leogane."

process_text(sent_2)
tokenize(sent_2)

#%%
#                       querying a corpus with gensim
# =============================================================================
"""
You'll use these data structures to investigate word trends and potential
interesting topics in your document set.

To get started, we have imported a few additional messy articles from
Wikipedia, which were preprocessed by:
    - lowercasing all words
    - tokenizing them
    - removing stop words and punctuation
These were then stored in a list of document tokens called articles.

You'll need to do some light preprocessing and then generate
the gensim dictionary and corpus
"""
articles = processed_files

# Import Dictionary
from gensim.corpora.dictionary import Dictionary

# Create a Dictionary from the articles: dictionary
dictionary = Dictionary(articles)

# Select the id for "computer": computer_id
computer_id = dictionary.token2id.get("computer")

# Use computer_id with the dictionary to print the word
print(dictionary.get(computer_id))

# Create a MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]

# Print the first 10 word ids with their frequency counts from the fifth document
print(corpus[4][:10])

#%%
for item in dictionary:
    print(item)

#%%
#                           Gensim bag-of-words
# =============================================================================
import itertools

# Save the fifth document: doc
doc = corpus[4]

# Sort the doc for frequency: bow_doc
bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)

# Print the top 5 words of the document alongside the count
for word_id, word_count in bow_doc[:8]:
    print(dictionary.get(word_id),word_count)

# Create the defaultdict: total_word_count
total_word_count = defaultdict(int)

for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count

list(itertools.chain.from_iterable(corpus))[:5]

# Create a sorted list from the defaultdict: sorted_word_count
sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)

# Print the top 5 words across all documents alongside the count
for word_id, word_count in sorted_word_count[:10]:
    print(dictionary.get(word_id), word_count)


#%%
#                           Tf-idf with Wikipedia
# =============================================================================
# Import TfidfModel
from gensim.models.tfidfmodel import TfidfModel

# Create a new TfidfModel using the corpus: tfidf
tfidf = TfidfModel(corpus)

# Calculate the tfidf weights of doc: tfidf_weights
tfidf_weights = tfidf[doc]

# Print the first five weights
print(tfidf_weights[:5])

# Sort the weights from highest to lowest: sorted_tfidf_weights
sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

# Print the top 5 weighted words
for term_id, weight in sorted_tfidf_weights[:10]:
    print(dictionary.get(term_id), weight)

#%%


# Most Common Words
# =============================================================================
count_dict = {}
for term_id, weight in corpus[0]:
    count_dict[dictionary.get(term_id)] = weight
    print(dictionary.get(term_id), weight)

Counter(count_dict).most_common(10)

