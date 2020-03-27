# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:08:59 2019

@author: smouz

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#%%
df = pd.read_csv('real_fake_news/fake_or_real_news.csv')

# Print the head of df
print(df.head())
print(df.columns)
print('DataFrame shape:', df.shape)
#%%

# =============================================================================
# CountVectorizer
# =============================================================================

# Create a series to store the labels: y
y = df['label']

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words='english',
                                   max_df=0.95,
                                   min_df=3,
                                   )

# Transform the training data using only the 'text' column values: count_train
count_train = count_vectorizer.fit_transform(X_train.values)

# Transform the test data using only the 'text' column values: count_test
count_test = count_vectorizer.transform(X_test.values)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])
# print(count_vectorizer.vocabulary_)
#%%

# =============================================================================
# TfidfVectorizer
# =============================================================================

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the training data: tfidf_train
tfidf_train = tfidf_vectorizer.fit_transform(X_train.values)

# Transform the test data: tfidf_test
tfidf_test = tfidf_vectorizer.transform(X_test.values)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])

tfidf_train.shape

#%%

# =============================================================================
# Inspect Vectors
# =============================================================================
# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))

"""
NOTE:
    columns are the same as show in `difference`. However, the values contained
    in the two DFs are not the same.
"""

#%%


# Import the necessary modules
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)

print("CountVec Accuracy:", np.round(score, 4))

#%%


# Create a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
tfidf_score = metrics.accuracy_score(y_test, pred)
print(tfidf_score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)

print("TfidfVec Accuracy:", np.round(tfidf_score, 4))
print("CountVec Accuracy:", np.round(score, 4))

#%%

# =============================================================================
# Improving the model
# =============================================================================
# Create the list of alphas: alphas
alphas = np.arange(0, 0.1, 0.01)

# Define train_and_predict()
def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier.fit(tfidf_train, y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    # Compute accuracy: score
    score = metrics.accuracy_score(y_test, pred)
    return np.round(score, 4)

# Iterate over the alphas and print the corresponding score
scores_array = np.zeros(alphas.shape[0])
for i, alpha in enumerate(alphas):
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    scores_array[i] = train_and_predict(alpha)
    print()

alpha_dict = {alpha: train_and_predict(alpha) for i, alpha in enumerate(alphas)}

print(sorted(alpha_dict.items(), key=lambda k: k[1], reverse=True)[0])

#%%

# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])
"""
Samples labeled `REAL` have the larger coefficients and samples labeled as
`FAKE` have smaller coefficients.

"""

#%%
nb_classifier.coef_[0].max()
nb_classifier.coef_[0].min()
nb_classifier.coef_[0].mean()
np.median(nb_classifier.coef_[0])

plt.hist(nb_classifier.coef_[0], bins=35)
plt.show()

class_labels[0]
