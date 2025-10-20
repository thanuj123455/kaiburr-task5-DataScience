Here is a detailed, professional README file for your project. You can copy and paste this directly into a README.md file in your Git repository.

-----

# 🚀 Consumer Complaint Text Classification

This project uses machine learning and Natural Language Processing (NLP) to classify consumer financial complaints from the [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database) into four specific categories.

The primary goal is to build an accurate and efficient multi-class classification model, compare the performance of several algorithms, and deploy the best model for predicting the category of new, unseen complaints.

*Target Categories:*

  * 0: Credit reporting, repair, or other
  * 1: Debt collection
  * 2: Consumer Loan
  * 3: Mortgage

-----

## 📊 Project Pipeline

This project follows a 6-step machine learning workflow:

1.  *Explanatory Data Analysis (EDA) and Feature Engineering:* Loaded the data, analyzed the Product column, and engineered the four target classes.
2.  *Text Pre-Processing:* Cleaned and normalized the raw complaint narratives to prepare them for vectorization.
3.  *Selection of Multi-Classification Models:* Chose a set of candidate models suitable for text data (Naive Bayes, Logistic Regression, Linear SVM, Random Forest).
4.  *Comparison of Model Performance:* Trained all models and compared them using the *Weighted F1-Score* to account for class imbalance.
5.  *Model Evaluation:* Performed a deep-dive analysis of the winning model using a classification report and confusion matrix.
6.  *Prediction:* Created a function to use the final, trained model to predict the category of new complaints.

-----

## ⚙ Installation & Setup

To run this project, clone the repository and install the required libraries.

bash
git clone [YOUR_REPOSITORY_URL]
cd [YOUR_PROJECT_DIRECTORY]


Install the necessary Python packages:

bash
pip install pandas scikit-learn nltk seaborn matplotlib joblib


You will also need to download the NLTK data for text preprocessing:

python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


-----

## 🔧 Methodology

### 1\. EDA and Feature Engineering

  * *Data Loading:* The dataset (rows.csv) was loaded into a pandas DataFrame.
  * *Column Selection:* Kept only the Product (target) and Consumer complaint narrative (feature) columns.
  * *Null Handling:* Dropped all rows where the Consumer complaint narrative was missing.
  * *Target Mapping:* The 18+ unique values in the Product column were mapped to our 4 target categories. Any products not in this map were dropped.
  * *Class Distribution:* An analysis showed a significant class imbalance, with Debt collection and Credit reporting... being the most frequent categories. This confirms that *Accuracy* is a poor metric, and *Weighted F1-Score* is the correct choice for evaluation.

### 2\. Text Pre-Processing

A custom function (preprocess_text) was created to clean the raw text. This pipeline performs the following steps on each complaint:

1.  *Remove Punctuation & Numbers:* Uses regex ([^a-zA-Z\s]) to keep only letters.
2.  *Lowercase:* Converts all text to lowercase.
3.  *Tokenization:* Splits the text into a list of individual words.
4.  *Stopword Removal:* Removes common English words (e.g., 'the', 'is', 'a') using nltk.corpus.stopwords.
5.  *Lemmatization:* Reduces words to their root form (e.g., 'running' $\rightarrow$ 'run') using nltk.WordNetLemmatizer.

### 3\. Model Selection

A scikit-learn Pipeline was used to bundle text vectorization and classification.

  * *Vectorization:* TfidfVectorizer was chosen to convert the cleaned text into a numerical matrix.
      * max_features=10000: Used the top 10,000 most important words.
      * ngram_range=(1, 2): Considered both single words (unigrams) and two-word phrases (bigrams), which helps capture more context (e.g., "credit report" vs. "credit" and "report" separately).
  * *Classifiers:* The following models were selected for comparison:
      * MultinomialNB
      * LogisticRegression
      * LinearSVC
      * RandomForestClassifier

-----

## 📈 Results: Model Comparison

All models were trained on the same data and evaluated on the test set. *Logistic Regression* provided the best balance of performance and training speed.

| Model | Weighted F1-Score |
| :--- | :--- |
| Naive Bayes | *0.87* |
| *Logistic Regression* | *0.90* |
| Linear SVM (SVC) | *0.90* |
| Random Forest | *0.90* |

### In-Depth Evaluation (Logistic Regression)

The chosen model (LogisticRegression) was evaluated, achieving a *0.90 weighted F1-score*.

*Classification Report:*


--- Classification Report (Logistic Regression) ---
                                  precision    recall  f1-score   support

Credit reporting, repair, or other       0.88      0.83      0.86      6376
                   Debt collection       0.89      0.93      0.91     17342
                     Consumer Loan       0.88      0.85      0.87      7755
                          Mortgage       0.95      0.95      0.95     10598

                          accuracy                           0.90     42071
                         macro avg       0.90      0.89      0.90     42071
                      weighted avg       0.90      0.90      0.90     42071


*Key Takeaways:*

  * The model is highly effective, with all categories achieving an F1-score of 0.86 or higher.
  * It is particularly strong at identifying Mortgage (0.95 F1) and Debt collection (0.91 F1) complaints.
  * The confusion matrix shows high accuracy on the diagonal. The most common errors are minor, such as confusing Consumer Loan with Mortgage, which is understandable given the semantic overlap.

-----

## 💾 Saved Model & Usage

The best-performing model (Logistic Regression + TfidfVectorizer pipeline) has been saved to the file best_complaint_classifier.joblib.

To use this model for predictions, you must load the model and also use the **exact same preprocess_text function** that was used in training.

### Example Prediction Script

python
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- 1. Define Helper Components (MUST be identical to training) ---

# Load NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define the category lookup
target_names = {
    0: 'Credit reporting, repair, or other',
    1: 'Debt collection',
    2: 'Consumer Loan',
    3. 'Mortgage'
}

def preprocess_text(text):
    """
    Cleans and prepares a single text string for the model.
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A).lower()
    tokens = word_tokenize(text)
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens 
        if word not in stop_words and len(word) > 2
    ]
    return ' '.join(processed_tokens)

# --- 2. Load the Model ---

model_filename = 'best_complaint_classifier.joblib'
best_model = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# --- 3. Create Prediction Function ---

def predict_complaint_category(text):
    """
    Takes a raw text complaint, cleans it, and predicts its category.
    """
    # 1. Pre-process the text
    cleaned_text = preprocess_text(text)
    
    # 2. Use the trained pipeline to predict
    prediction_code = best_model.predict([cleaned_text])[0]
    
    # 3. Look up the category name
    return target_names[prediction_code]

# --- 4. Make a Prediction ---

new_complaint = """
I am being harassed by a debt collector. They call my work 
and my cell phone 10 times a day for a debt I already paid. 
I have told them to stop calling me.
"""

prediction = predict_complaint_category(new_complaint)
print(f"\nNew Complaint:\n{new_complaint}")
print(f"\nPrediction: {prediction}")


*Expected Output:*


Model loaded from best_complaint_classifier.joblib

New Complaint:
I am being harassed by a debt collector. They call my work 
and my cell phone 10 times a day for a debt I already paid. 
I have told them to stop calling me.

Prediction: Debt collection
