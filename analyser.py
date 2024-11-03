# analyser.py
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
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

# 3. Data Distribution Visualizations
plt.figure(figsize=(12, 5))

# Distribution of Sentiment Classes
plt.subplot(1, 2, 1)
sns.countplot(x='Sentiment', data=df, palette='Set2')
plt.title('Distribution of Sentiment Classes')
plt.xlabel('Sentiment')
plt.ylabel('Count')

# Distribution of Scores
plt.subplot(1, 2, 2)
sns.histplot(df['Score'], bins=5, kde=True, color='skyblue')
plt.title('Distribution of Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 4. Sentiment Analysis Model Development
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# 5. Model Evaluation
X_test_vectorized = vectorizer.transform(X_test)
predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Additional metrics
print("Classification Report:\n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# 6. Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# 7. Save the model and vectorizer
with open('model/sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('model/vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")
