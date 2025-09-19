
````markdown
# Restaurant Review Sentiment Analysis

A Python-based Natural Language Processing (NLP) project to classify restaurant reviews as **Positive** or **Negative** using a **Naive Bayes classifier**. The project uses a dataset of 1000 restaurant reviews and implements data preprocessing, model training, evaluation, and prediction.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Data Preprocessing](#data-preprocessing)  
- [Model Building](#model-building)  
- [Evaluation](#evaluation)  
- [Usage](#usage)  
- [Technologies](#technologies)  
- [Results](#results)  

---

## Project Overview

This project classifies restaurant reviews into positive or negative sentiment. It uses Natural Language Processing techniques to clean and process the text data, and a **Multinomial Naive Bayes classifier** for training. The trained model can predict the sentiment of any new review.

---

## Dataset

The dataset `Restaurant_Reviews.tsv` contains 1000 restaurant reviews with two columns:

| Column | Description |
|--------|-------------|
| Review | Text of the restaurant review |
| Liked  | Sentiment label: 1 (Positive), 0 (Negative) |

The dataset is stored in TSV format and can be found in `Colab Notebooks/Datasets/Restaurant_Reviews.tsv`.

---

## Data Preprocessing

The preprocessing steps include:

1. **Cleaning text**: Remove special characters and punctuation.  
2. **Lowercasing**: Convert all words to lowercase.  
3. **Tokenization**: Split reviews into individual words.  
4. **Stopwords removal**: Remove common English stopwords.  
5. **Stemming**: Reduce words to their root forms using `PorterStemmer`.  
6. **Bag of Words**: Convert processed reviews into a matrix of token counts using `CountVectorizer`.

---

## Model Building

- **Classifier**: `MultinomialNB` (Naive Bayes)  
- **Train-Test Split**: 80% training, 20% testing  
- **Hyperparameter Tuning**: Tested different alpha values to achieve the best accuracy.  

**Best Parameters**:  
- Alpha: 0.2  
- Accuracy: 78.5%

---

## Evaluation

Model evaluation metrics:

- **Accuracy**: 78.5%  
- **Precision**: 0.76  
- **Recall**: 0.79  

**Confusion Matrix**:

|               | Predicted Negative | Predicted Positive |
|---------------|-----------------|-----------------|
| Actual Negative | 72              | 25              |
| Actual Positive | 22              | 81              |

Visualization of the confusion matrix is done using **Seaborn** heatmap.

---

## Usage

1. **Mount Google Drive (if using Colab)**:

```python
from google.colab import drive
drive.mount('/content/drive/')
````

2. **Load Dataset**:

```python
import pandas as pd
df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Datasets/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
```

3. **Train Model**:

* Preprocess the data.
* Build Bag of Words model.
* Train Naive Bayes classifier.
* Tune hyperparameters for best accuracy.

4. **Predict New Reviews**:

```python
sample_review = 'The food is really good here.'

if predict_sentiment(sample_review):
    print('This is a POSITIVE review.')
else:
    print('This is a NEGATIVE review!')
```

---

## Technologies

* Python 3.x
* Google Colab
* Pandas, NumPy
* NLTK (Stopwords, PorterStemmer)
* Scikit-learn (MultinomialNB, CountVectorizer, train\_test\_split)
* Matplotlib, Seaborn

---

## Results

The trained model can correctly classify reviews with an accuracy of **78.5%**, and performs reasonably well on unseen reviews:

* `"The food is really good here."` → **Positive**
* `"Food was pretty bad and the service was very slow."` → **Negative**
* `"The food was absolutely wonderful, from preparation to presentation, very pleasing."` → **Positive**

---


