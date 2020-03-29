# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 21:14:56 2020

@author: smouz

"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn import metrics

#%%

df = pd.read_csv('real_fake_news/fake_or_real_news.csv')

# Print the head of df
print(df.head())
print(df.columns)
print('DataFrame shape:', df.shape)

# =============================================================================
# CountVectorizer
# =============================================================================

# Create a series to store the labels: y
# place 1 where news is FAKE
y = np.where(df['label']=='FAKE', 1, 0)
X = df['text']

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=11)


def tokenize(text):
    """
    Replace `url` with empty space "".
    Tokenize and lemmatize input `text`.
    Converts to lower case and strips whitespaces.


    Returns:
    --------
        dtype: list, containing processed words
    """

    lemm = WordNetLemmatizer()

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "")

    # load stopwords
    stop_words = stopwords.words("english")

    # remove punctuations (retain alphabetical and numeric chars) and convert to all lower case
    # tokenize resulting text
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", ' ', text.lower().strip()))

    # drop stop words
    no_stops = [word for word in tokens if word not in stop_words]

    # lemmatize and remove stop words
    lemmatized = [lemm.lemmatize(word) for word in tokens if word not in stop_words]

    return lemmatized

def show_metrics(y_true, y_pred):
    print('-'*75)
    print('Accuracy:', np.round(metrics.accuracy_score(y_true, y_pred), 4))
    print('F1-Score:', np.round(metrics.f1_score(y_true, y_pred), 4))
    print('-'*75)


def baseline_model(clf):
    """
    Baseline model for comparison.

    Count Vectorizer pipeline trains a given classifier on training
    subset.

    Returns:
    --------
        Sklearn Pipeline object

    """
    # Initialize a CountVectorizer object: count_vectorizer
    count_vectorizer = CountVectorizer(stop_words='english',
                                       tokenizer=tokenize,
                                       dtype=np.int32,
                                       max_df=0.98,
                                       min_df=2,
                                       ngram_range=(1, 2),
                                      )

    classifier = Pipeline([
        ('count_vec', count_vectorizer),
        ('clf', clf)
        ])

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print('\nBaseline metrics:')
    show_metrics(y_test, y_pred)

    return classifier


def evaluate_model(clf):
    """
    Count Vectorizer pipeline trains a given classifier on training
    subset.

    Returns:
    --------
        Sklearn Pipeline object

    """
    # Initialize a CountVectorizer object: count_vectorizer
    count_vectorizer = CountVectorizer(stop_words='english',
                                       tokenizer=tokenize,
                                       dtype=np.int32,
                                       max_df=0.98,
                                       min_df=2,
                                       ngram_range=(1, 2),
                                      )

    classifier = Pipeline([
        ('count_vec', count_vectorizer),
        ('clf', clf)
        ])

    classifier.fit(X_train, y_train)

    return classifier
#%%
# baseline model
clf = LogisticRegression(C=0.1, max_iter=200)
baseline_model(clf)


#%%
# =============================================================================
# Blending Classifiers
# =============================================================================
# define classifier for blending
n_jobs = 7
classifiers = [
    ('bagging', BaggingClassifier(
        n_estimators=50,
        max_samples=0.5,
        max_features=0.5,
        bootstrap_features=True,
        random_state=11,
        n_jobs=n_jobs
        )),
    ('random_forest_gini', RandomForestClassifier(
        n_estimators=50,
        # max_features=0.6,
        # max_samples=0.8,
        n_jobs=n_jobs,
        random_state=11
        )),
    ('random_forest', RandomForestClassifier(
        n_estimators=50,
        # max_features=0.6,
        # max_samples=0.8,
        n_jobs=n_jobs,
        criterion='entropy',
        random_state=11
        )),

    ('log_reg', LogisticRegression(
        C=0.1,
        n_jobs=n_jobs,
        max_iter=200,
        random_state=11
        )),

    ('multinomial_nb', MultinomialNB(
        alpha=0.03,
        )),

    ('svm', SVC(
        C=0.1,
        kernel='linear',
        probability=True,
        random_state=11
        ))

    ]


#### Blend Probabilities
# using the pipeline from above, train each classifer on Train Subset
# predict probabilities on Test Subset using each classifier
# blend probabilities to form new predictions
n_clf = len(classifiers)

preds = np.zeros(X_test.shape[0])
for clf in classifiers:
    print('\nTraining classifier...')
    print(clf[0])
    model = evaluate_model(clf[1])

    print('\nBlending predictions...')
    preds = np.add(model.predict_proba(X_test)[:, 1], preds)

print("\nSaving predictions...")
np.save('test_preds_matrix', preds)

# average all predictions
# preds_blend = np.mean(preds[:, :n_clf-1], axis=1)
preds_blend = preds / n_clf

# where prob is greater than 50%, replace with 1
# probability threshold may be varied, plus or minus 0.10
final_preds = np.where(preds_blend > 0.55, 1, 0)

# compute accuracy
print('-'*75)
print('Blended metrics:\n')
show_metrics(y_test, final_preds)


