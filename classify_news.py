# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 21:14:56 2020

@author: smouz

"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


