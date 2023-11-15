import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from wordcloud import WordCloud
from collections import Counter

# Load the dataset
df = pd.read_csv("C:/Users/HP/Desktop/spam.csv", encoding='latin', usecols=['v1', 'v2'])
df.columns = ['label', 'text']

# Data preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    stemmer = SnowballStemmer("english")
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text

df['text'] = df['text'].apply(clean_text)

# Split the data into training and testing sets
X = df['text']
y = df['label'].map({'spam': 1, 'ham': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43, stratify=y)

# Feature extraction
tfidf = TfidfVectorizer(ngram_range=(1, 3))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model training and evaluation
mnb = MultinomialNB()
params = {'alpha': [0.1, 0.5, 1, 5, 10]}
rcv = RandomizedSearchCV(mnb, params, scoring='accuracy', cv=10, n_jobs=-1, random_state=3, verbose=3)
rcv.fit(X_train_tfidf, y_train)

print('Best Accuracy:', rcv.best_score_, 'Best Parameters:', rcv.best_params_)

mnb_tfidf = MultinomialNB(alpha=0.5)
mnb_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = mnb_tfidf.predict(X_test_tfidf)

print(classification_report(y_test, y_pred_tfidf))

# Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred_tfidf)
fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True, show_normed=True, colorbar=True, class_names=["ham", "spam"], cmap='RdGy_r')
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_tfidf)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color="green", lw=2, label=f"ROC curve (area = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.legend(loc="lower right")
plt.show()
