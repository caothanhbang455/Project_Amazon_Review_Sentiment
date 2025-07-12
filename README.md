# English Text Classification - Amazon Review Sentiment

This project builds a text classification pipeline to detect the language and sentiment of English product reviews, using a mix of natural language processing (NLP) and machine learning techniques.

**Author**: Cao Thanh Bằng  

---

## Project Objective

The goal of this project is to build a robust pipeline to preprocess, vectorize, and train a binary classification model using logistic regression on English-language review texts. The model predicts sentiment (positive or negative) based on the content of product reviews.

---

## Dataset

- **Source**: Amazon product reviews
- **Key fields used**:
  - `reviewText`: the body of the review written by a user
  - `overall`: the rating score from 1 to 5, used to create a binary `label`:
    - 1 → Negative
    - 5 → Positive
- Total records: several thousand entries

---
## Running

```bash
# Clone this repository
git clone https://github.com/caothanhbang455/Project_predict_India_rental_house.git

# Open the notebook
jupyter notebook FullName_CK_K304_HV.ipynb
```

## Key Steps

### 1. Preprocessing

- Remove punctuation using `string.punctuation`
- Convert text to lowercase
- Remove numbers using regex (`\d+`)
- Normalize whitespace
- Apply lemmatization using `WordNetLemmatizer`
- Remove stopwords using `nltk.corpus.stopwords`
- Detect and filter by language using `langdetect` (only English texts are retained)

### 2. Feature Extraction

- Apply TF-IDF vectorization using `TfidfVectorizer`
- Limit vocabulary to the top 900 most frequent features
- Automatically remove English stopwords via `stop_words='english'`

### 3. Modeling

- Split the dataset into train and test sets (80/20)
- Train a logistic regression model using `scikit-learn`
- Optionally tune the classification threshold manually

### 4. Evaluation

- Use the following metrics for evaluation:
  - Accuracy score
  - Confusion matrix
  - Classification report (precision, recall, F1-score)

### 5. Model Export

The pipeline (vectorizer + model) is exported using `pickle`:

```python
import pickle
pickle.dump(_pipeline, open('model_checkpoints/model.pkl', 'wb'))


