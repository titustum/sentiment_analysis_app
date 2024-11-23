import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score

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

# Pie Chart for Sentiment Distribution
plt.subplot(1, 3, 1)
sentiment_counts = df['Sentiment'].value_counts()
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title('Sentiment Distribution')

# Distribution of Sentiment Classes
plt.subplot(1, 3, 2)
sns.countplot(x='Sentiment', data=df, palette='Set2')
plt.title('Distribution of Sentiment Classes')
plt.xlabel('Sentiment')
plt.ylabel('Count')

# Boxplot for Review Lengths
df['Text Length'] = df['Text'].apply(len)
plt.subplot(1, 3, 3)
sns.boxplot(x='Sentiment', y='Text Length', data=df, palette='Set2')
plt.title('Review Length Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Review Length')

plt.tight_layout()
plt.show()

# 4. Word Cloud for Frequent Words
text = ' '.join(df['Text'])
wordcloud = WordCloud(stopwords='english', background_color='white', width=800, height=400).generate(text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Reviews')
plt.show()

# 5. Time Series Plot for Sentiment Trends
df['Time'] = pd.to_datetime(df['Time'], unit='s')
df.set_index('Time', inplace=True)
df_resampled = df.resample('M').apply(lambda x: x['Sentiment'].value_counts().idxmax())

plt.figure(figsize=(12, 6))
df_resampled.value_counts().plot(kind='line', marker='o', color=['skyblue', 'salmon'])
plt.title('Sentiment Trend Over Time')
plt.xlabel('Month')
plt.ylabel('Sentiment Count')
plt.xticks(rotation=45)
plt.show()

# 6. Sentiment Analysis Model Development
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# List of models to compare
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier()
}

# 7. Train and Evaluate Each Model
for model_name, model in models.items():
    # Train model
    model.fit(X_train_vectorized, y_train)
    
    # Make predictions
    predictions = model.predict(X_test_vectorized)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f'\nModel: {model_name}')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print("Classification Report:\n", classification_report(y_test, predictions))
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

# 8. Save the best model (Naive Bayes as an example)
best_model = models['Naive Bayes']
with open('model/sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('model/vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")
