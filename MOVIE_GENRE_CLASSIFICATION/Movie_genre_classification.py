# Import libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
print("Loading dataset...")
df = pd.read_csv("genre_classification.csv")  # Replace with your dataset name

# Display first few rows of the dataset
print("Dataset preview:")
print(df.head())

# Check for missing values
print("\nChecking for missing values...")
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Preprocessing text
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

print("\nPreprocessing text data...")
df['cleaned_plot'] = df['plot'].apply(preprocess_text)

# TF-IDF vectorization
print("\nVectorizing text data using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_plot']).toarray()

# Encoding target variable
print("\nEncoding target variable...")
y = df['genre']

# Splitting the data into train and test sets
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
print("\nTraining the Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Making predictions
print("\nMaking predictions...")
y_pred = model.predict(X_test)

# Model evaluation
print("\nModel evaluation:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
import joblib
print("\nSaving the trained model...")
joblib.dump(model, "movie_genre_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\nModel training and evaluation complete!")
