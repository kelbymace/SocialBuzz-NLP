# SocialBuzz NLP: Binary Sentiment Classification

This project builds a binary sentiment classifier on social media text using a Kaggle dataset of posts.

## Project Summary
- Task: Binary text classification (`Final Sentiment`)
- Data source: Kaggle social media posts dataset
- Input text column: `Text_lemma` (after preprocessing)
- Output: Predicted sentiment labels and model comparison

## Dataset Files
- `sentimentdataset.csv`: Primary dataset used for training and evaluation
- `sentiment_with_predictions.csv`: Dataset with model predictions

## Label Consolidation (Key Step)
The original pre-labeled sentiment field contained a very large number of categories (over 100), which was not practical to model reliably with a dataset of about 700 rows.

To make the problem learnable:
- Zero-shot classification was applied to both post text and sentiment labels
- Labels were mapped into three consolidated classes: `positive`, `neutral`, `negative`
- The neutral class represented fewer than 4% of samples, so neutral rows were removed
- Final modeling was run as binary classification (`positive` vs `negative`)

## Preprocessing Pipeline
Text data was cleaned and normalized before modeling. Main steps included:
- Lowercasing and text normalization
- Tokenization
- Stop word removal
- Lemmatization
- Regex-based cleanup / punctuation handling
- Preparation of final cleaned text field for vectorization

## Feature Engineering
Two text representation approaches were used:

1. TF-IDF
- Sparse bag-of-words style representation
- Captures term importance across documents

2. GloVe Embeddings
- Pretrained dense word vectors (`glove-twitter-100`)
- Document vectors created by averaging token embeddings
- This approach produced the best overall performance in this project when paired with Logistic Regression

## Models Trained
The following binary classifiers were tested:
- Logistic Regression
- Linear SVM (`LinearSVC`)
- Naive Bayes (GaussianNB for dense GloVe vectors)

## Evaluation
Models were evaluated with standard classification metrics, including:
- Classification report (precision, recall, F1-score)
- Confusion matrix
- ROC-AUC (for binary classification)

## Key Result
GloVe + Logistic Regression performed best among the tested models.

## Repository Structure
- `SocialBuzz NLP.ipynb`: Main notebook with preprocessing, feature engineering, training, and evaluation
- `sentimentdataset.csv`: Input data
- `sentiment_with_predictions.csv`: Output with predictions

## Reproducibility Notes
To run the notebook end-to-end, install common NLP/ML dependencies such as:
- `pandas`, `numpy`
- `scikit-learn`
- `nltk`
- `gensim`
- `matplotlib`, `seaborn`

If using GloVe through `gensim.downloader`, vectors are downloaded on first run and cached locally.

## Future Improvements
- Cross-validated hyperparameter tuning (especially for Logistic Regression/SVM)
- Threshold tuning for class-specific precision/recall tradeoffs
- Error analysis on misclassified posts
- Comparison with transformer-based sentence embeddings on the same split
