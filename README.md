### NATURAL LANGUAGE PROCESSING (NLP)

## Project Description

_Detecting Similarity and Key Terms in Brief Clinical Texts._

**Goal**: Apply traditional NLP techniques to analyze semantic similarity and extract key terms from brief clinical texts using BoW, TF-IDF, and cosine similarity.

# 📋 Table of Contents

- [🚀 Features](#features)
- [📦 Requirements](#requirements)
- [🚀 How to Run](#how-to-run)
- [📚 NLP Definition](#nlp-definition)
- [📚 BoW](#bow)
- [📚 TF-IDF](#tf-idf)
- [📈 Key Learnings](#key-learnings)

## Features

- **Automatic preprocessing** of clinical text
- **Vectorization** with BoW and TF-IDF
- **Similarity calculation** using cosine similarity
- **Visualization** with heatmaps and graphs
- **Key term extraction**

## Requirements

- **Languaje**: Python 3.8+
- **ML & NLP**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## How to Run

- Clone this repository or download the source files.

```bash
1. Visit [Google Colab](https://colab.research.google.com/)
2. Upload your `.ipynb` file or open from GitHub
3. Run cells with Ctrl+Enter
```
## NLP Definition?

**Natural language processing (NLP)** According to Wikipedia NLP is an interdisciplinary subfield of computer science and artificial intelligence. It is primarily concerned with providing computers the ability to process and understand human language, and to generate human language in response. 

## What is this Notebook about?
Using Natural Language Processing, this notebook will evaluate similarity between simulated clinical notes and interpret their semantic relationship through basic metrics and visualizations.

The focus in NLP is on the foundational step of feature extraction, which is crucial for preparing text data for analysis. The Basic NLP transformation is as follow:

_Original Sentence:  "The patients are showing improved symptoms"_

1. Tokenize:  ["The@", "patients3", "are", "showing", "improved", "symptoms"]
2. Cleaning and normalization: ["the", "patients", "are", "showing", "improved", "symptoms"]
3. Stopwords: ["patients", "showing", "improved", "symptoms"]
4. Lemmatize: ["patient", "show", "improve", "symptom"]
5. Vectorize: [1, 1, 1, 1, 0, 0]  # Numerical representation

## BoW
The **Bag-of-Words (BoW)** will be used to represents text as an unordered collection of words, ignoring grammar and word order but keeping multiplicity.

**How it works**:
- Creates a vocabulary of all unique words in the corpus
- Each document is represented as a vector of word counts
- Simple but effective for basic text classification

**Example**:

```python
# Bag-of-Words (BoW) Implementation in Python
from from sklearn.feature_extraction.text import CountVectorizer

# Sample corpus of clinical notes
corpus = [
    "Patient presents with fever and cough",
    "Patient has high fever and headache", 
    "Patient reports cough and fatigue",
    "Patient shows fever cough and sore throat"
]

def bow_sklearn(corpus):
    # Create CountVectorizer instance
    vectorizer = CountVectorizer()
    
    # Fit and transform the corpus
    X = vectorizer.fit_transform(corpus)
    
    # Get feature names (vocabulary)
    feature_names = vectorizer.get_feature_names_out()
    
    # Create DataFrame for better visualization
    bow_df = pd.DataFrame(X.toarray(), 
                         columns=feature_names,
                         index=[f'Doc_{i+1}' for i in range(len(corpus))])


# Output:
Bag-of-Words Matrix:
        and  cough  fatigue  fever  has  headache  high  patient  presents  reports  shows  sore  throat  with
Doc_1    1      1        0      1    0         0     0        1         1        0      0     0       0     1
Doc_2    1      0        0      1    1         1     1        1         0        0      0     0       0     0
Doc_3    1      1        1      0    0         0     0        1         0        1      0     0       0     0
Doc_4    1      1        0      1    0         0     0        1         0        0      1     1       1     0
```
## TF-IDF
The **TF-IDF (Term Frequency-Inverse Document Frequency)** is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. It combines two metrics: **Term Frequency (TF)** and **Inverse Document Frequency (IDF)**.

## Practical Example

```python
# TF-IDF (Term Frequency-Inverse Document Frequency) in Python
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Get feature names and TF-IDF matrix
feature_names = vectorizer.get_feature_names_out()
tfidf_matrix = X.toarray()

#Output:
📄 Doc_1:
------------------------------
  presents: 0.4807
  with: 0.4807
  cough: 0.3841
  fever: 0.3841
  and: 0.3333
  patient: 0.3333

📄 Doc_2:
------------------------------
  has: 0.4807
  headache: 0.4807
  high: 0.4807
  fever: 0.3841
  and: 0.3333
  patient: 0.3333

📄 Doc_3:
------------------------------
  fatigue: 0.4807
  reports: 0.4807
  cough: 0.3841
  and: 0.3333
  patient: 0.3333

  📄 Doc_4:
------------------------------
  sore: 0.4163
  throat: 0.4163
  shows: 0.4163
  cough: 0.3333
  fever: 0.3333
  and: 0.2887
  patient: 0.2887
```

## Key Learnings

This project solidified my understanding of how Natural Language Processing (NLP) transcends theoretical exercise to become a critical technology with real-world impact. By using techniques like BoW, TF-IDF, and cosine similarity to analyze clinical symptoms, I saw firsthand how NLP can be a powerful tool in healthcare which involves every human life. The ability to process, understand, and find patterns in medical text automatically is not just an technical achievement; it's a stepping stone towards building systems that can assist in diagnostics, manage public health trends, and improve patient care for everyone. This is where technology truly fulfills its purpose: when it serves humanity in its most fundamental needs.