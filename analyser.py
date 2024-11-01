# analyser.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Data Extraction
df = pd.read_csv('Reviews.csv')
print(df.head())

# 2. Data Cleaning
df.dropna(subset=['Text'], inplace=True)
df['Text'] = df['Text'].str.replace(r'[^\w\s]', '', regex=True).str.lower()

# Binning scores into positive and negative categories
df['Sentiment'] = df['Score'].apply(lambda x: 'positive' if x > 3 else 'negative')
X = df['Text']
y = df['Sentiment']

# 3. Sentiment Analysis Model Development
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# 4. Model Evaluation
X_test_vectorized = vectorizer.transform(X_test)
predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Additional metrics
print("Classification Report:\n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# 5. Save the model and vectorizer
with open('model/sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('model/vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")