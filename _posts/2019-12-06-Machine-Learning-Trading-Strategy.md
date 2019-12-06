---
layout: post
Title: Machine Learning Trading Strategy
Subtitle: Using Machine Learning to Generate a Trading Strategy
---

## Introduction
To perform a machine learning on tweets, the data needs to be cleaned, tokenised and vectorised. We will show the steps we have taken to run the machine learning algorithm on our data.
### Loading Libraies
These are the libraries that we used for the project. They can be installed using [pip](https://pypi.org/project/pip/) or [Anaconda](https://www.anaconda.com/).
```python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
```
## Data Cleaning
We will use [Dask](https://dask.org/) for data cleaning, which allows our Pandas workflow to be scaled and parallelised. Dask uses existing Python APIs and data structures to make it easy to switch between Numpy, Pandas, Scikit-learn to their Dask-powered equivalents.
### Loading the Data
We imported the csv file we collected using Tweepy as a Dask dataframe. During the data collection process, some of the rows in the csv file might be corrupted due to connection issues. We used the python csv engine, which is slower but can tolerate more errors, and we set “error_bad_lines” to False, so that problematic rows will be skipped instead of causing an exception to be thrown.

Sometimes, dask might guess the wrong data type of our columns, and we can adjust the dtype to fix this.
```python
df = dd.read_csv(
    "data-streaming-tweets.csv",
    names=["user", "followers", "time", "text"],
    engine="python",
    encoding="utf-8",
    error_bad_lines=False,
    dtype={"followers": "object"},
)
```
### Text Processing and Tokenisation
Tokenization is the task of chopping the Tweets up into pieces, called tokens, perhaps at the same time throwing away certain characters, such as punctuation. 

For our tokenisation, we used the Tweet Tokenizer in [NLTK](https://www.nltk.org/) to tokenise our tweets. NLTK is a leading platform for building Python programs to work with human language data.

By setting “preserve_case” to False, the tokenizer converts our text to lower case, and setting “strip_handles” to True removes all Twitter usernames. We also removed all non-alphabetical tokens, because they add too much noise to our results.
```python
tknzr = nltk.tokenize.TweetTokenizer(preserve_case=False, strip_handles=True)
def tokenize(text):
    tokens = tknzr.tokenize(str(text))
    return list(filter(lambda word: word.isalpha(), tokens))
```
### Stop Words Removal
A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that our machine learning algorithm should ignore, since they are of little value when trying to understand a Tweet. We would not want these words taking up space in our database, or taking up valuable processing time, so they will be removed.

We used the corpus from NLTK to remove stop words. We also added meaningless words we found in our dataset that affect the results into the list.
```python
stop_words = nltk.corpus.stopwords.words("english")
stop_words.extend(["bitcoin", "btc", "http", "https", "rt"])
def remove_stopwords(words):
    filtered = filter(lambda word: word not in stop_words, words)
    return list(filtered)
```
### Lemmatisation
In the last step of data cleaning, we lemmatised the words. Lemmatisation is the algorithmic process of determining the root of a word based on its intended meaning. Unlike stemming, lemmatisation depends on correctly identifying the intended part of speech and meaning of a word in a sentence, as well as within the larger context surrounding that sentence, such as neighboring sentences or even an entire document.

We discovered that the NLTK tokenizer does not work well when the correct part-of-speech tags are not given together with the word. Instead, we will use [spaCy](https://spacy.io/), which determines the part-of-speech tag by default and assigns the corresponding lemma.

spaCy is a free, open-source library for advanced Natural Language Processing (NLP) in Python. It is designed specifically for production use and helps build applications that process and “understand” large volumes of text. 
```python
nlp = spacy.load("en_core_web_sm")
def lemmatize(text, nlp=nlp):
    doc = nlp(" ".join(text))
    lemmatized = [token.lemma_ for token in doc]

    return lemmatized
```
### Computing and Saving
We create a function to combine all these steps together, and use the map_partitions() function to tell Dask to run it on every row. Dask delays computation to improve performance. To obtain our results, we have to call the compute() function. We then save the results in the [Apache Parquet](https://parquet.apache.org/) format.

Apache Parquet is a columnar storage format from the Hadoop ecosystem. It provides efficient data compression and encoding schemes to handle large datasets with high performance requirements.
```python
def clean_text(df):
    df["cleaned"] = df.text.map(tokenize).map(remove_stopwords).map(lemmatize)
    return df
tokens = df.map_partitions(clean_text).compute()
tokens.to_parquet("tokens.parquet", engine="pyarrow")
```
## Machine Learning
We will try to train the machine learning algorithm to predict how bitcoin prices will change in the next hour. Every tweet will be classified into two groups, rise or fall. Since it is unlikely that we have enough information in a single tweet to predict prices, we will average the results of all the tweets in an hour. We hope that this will allow us to predict bitcoin prices to a reasonable degree of accuracy.

Most of the algorithms we used below are part of the [Scikit-learn](https://scikit-learn.org) library. Scikit-learn is a free software machine learning library for the Python programming language.
### Loading Tokens
We load the Parquet file generated in the data cleaning step, and group the time by hours using the dt.floor(“H”) function.
```python
df = pd.read_parquet("/home/noel/Desktop/tokens.parquet",
                     columns=["time", "cleaned"], engine="pyarrow")
df.dropna(inplace=True)
df["date"] = pd.to_datetime(
    df["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
df.dropna(inplace=True)
df["date"] = df["date"].dt.floor("H")
```
### Loading Bitcoin Price Data
Next, we downloaded the hourly bitcoin price from [CryptoDataDownload](https://www.cryptodatadownload.com). We used data from the Coinbase exchange, since it is one of the largest bitcoin exchanges in the world.
```python
price = pd.read_csv("~/Desktop/Coinbase_BTCUSD_1h.csv")
price.sort_values(by="Date", inplace=True)
price.reset_index(drop=True, inplace=True)
price["Date"] = pd.to_datetime(price["Date"], format=r"%Y-%m-%d %I-%p")
```
### Categorising the Hours
Each hour is then categorised into two groups. 0 represents a fall in the price, while 1 represents a rise. This creates a binary classification problem that can be used to train the machine learning algorithm.
```python
price["nClose"] = price["Close"].shift(-1)
price["Return"] = price["nClose"] / price["Close"] - 1
price.dropna(inplace=True)

price["cat"] = np.where(price["Return"] > 0, 1, 0)
```
### Merging Data
We remove the unnecessary data in the DataFrame and merge the price categories with the tokens.
```python
ret = price[["Date", "cat"]].copy()
ret.columns = ["date", "ret"]
del price
df = pd.merge(df, ret, on="date")
df.dropna(inplace=True)
```
### Text Data Vectorisation
Next, we convert our tokens into Bag of Words or Tf-idf. We create a dummy function to disable the built-in preprocessing and tokenisation of the CountVectorizer and TfidfVectorizer.
```python
def dummy(x):
    return x
cv = CountVectorizer(tokenizer=dummy, lowercase=False, preprocessor=dummy, min
_df=100)  # Or TfidfVectorizer
bow = cv.fit_transform(df["cleaned"])
```
### Training the Machine Learning Algorithms
The data is split into a test and training set using NLTK’s train_test_split in the ratio of 2:1. We then train the algorithms using the fit() function, and validated the resulting using the predict() and classification_report() functions.

Since different algorithms have differnt strengths and weaknesses, we will test different algorithms on our data. The algorithms tested in the code below are [Naive Bayes](scikit-learn.org/stable/modules/naive_bayes.html), [XGBoost](https://xgboost.readthedocs.io/en/latest/), [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) and [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
```python
X_train, X_test, y_train, y_test = train_test_split(
    bow, df, test_size=0.33, random_state=42
)

# NB
print("Naive Bayes")
nb = MultinomialNB()
nb.fit(X_train, y_train.ret)

nb_res = nb.predict(X_test)
print(classification_report(y_test.ret, nb_res))

# XGB
print("XGBoost")
model = XGBClassifier()
model.fit(X_train, y_train.ret)

xgb_res = model.predict(X_test)
print(classification_report(y_test.ret, xgb_res))

# Logistic Regression
print("Logistic Regression")
lr = LogisticRegression(solver="saga")
lr.fit(X_train, y_train.ret)

lr_res = lr.predict(X_test)
print(classification_report(y_test.ret, lr_res))

# Support Vector Machine

print("Support Vector Machine")
svm = LinearSVC()
svm.fit(X_train, y_train.ret)

svm_res = svm.predict(X_test)
print(classification_report(y_test.ret, svm_res))

# Random Forest
print("Random Forest")
rf = RandomForestClassifier()
rf.fit(X_train, y_train.ret)

rf_res = rf.predict(X_test)
print(classification_report(y_test.ret, rf_res))
```
This will generate a classification report for each algoritm, allow us to compare the different algorithms. One example is shown below:
```
Random Forest
              precision    recall  f1-score   support

           0       0.57      0.57      0.57    276584
           1       0.58      0.57      0.57    280866

    accuracy                           0.57    557450
   macro avg       0.57      0.57      0.57    557450
weighted avg       0.57      0.57      0.57    557450
```
## Conclusion
For us, the random forest classifier produced the best results. However, this might differ based on your dataset. Machine learning is more of an art than a science, and it takes a lot of trial and error to find the best parameters.

We you found this post helpful!
