# analyser.py
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# 1. Data Extraction
df = pd.read_csv('Reviews.csv')
print(df.head())

# 2. Data Cleaning
df.dropna(subset=['Text'], inplace=True)
df['Text'] = df['Text'].str.replace(r'[^\w\s]', '', regex=True).str.lower()

# 3. Binning scores into positive, negative, and neutral categories
df['Sentiment'] = df['Score'].apply(lambda x: 'positive' if x > 3 else ('negative' if x < 3 else 'neutral'))
X = df['Text']
y = df['Sentiment']

# 4. Data Distribution Visualization

# Distribution of Sentiment Classes
plt.figure(figsize=(12, 6))
sns.countplot(x='Sentiment', data=df, palette='Set2')
plt.title('Distribution of Sentiment Classes')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Distribution of Scores
plt.figure(figsize=(12, 6))
sns.histplot(df['Score'], bins=5, kde=True, color='skyblue')
plt.title('Distribution of Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 5. Word Cloud for Text Data (Most Frequent Words in the Reviews)
plt.figure(figsize=(10, 8))
wordcloud = WordCloud(stopwords='english', background_color='white', width=800, height=400).generate(' '.join(df['Text'][:100]))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Review Text')
plt.axis('off')
plt.show()

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Feature Extraction
vectorizer = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 8. Model Development - Naive Bayes, Logistic Regression, SVM, Random Forest
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(),
    # 'Random Forest': RandomForestClassifier(), # Computationally expensive, especially when dealing with large datasets 
    # 'SVM': SVC(probability=True) # Take a long time to train, especially on large datasets
}

# 9. Model Training and Evaluation
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(X_test_vectorized)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
    
    # Classification report and confusion matrix
    print(f"\n{model_name} Classification Report:\n", classification_report(y_test, predictions))
    print(f"{model_name} Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    
    # Confusion Matrix Heatmap
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'], 
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()
    
    # ROC Curve for SVM and Random Forest (requires probability prediction)
    # if model_name in ['SVM', 'Random Forest']:
    #     fpr, tpr, _ = roc_curve(y_test.apply(lambda x: 0 if x == 'negative' else (1 if x == 'neutral' else 2)),
    #                              model.predict_proba(X_test_vectorized)[:, 1])
    #     roc_auc = auc(fpr, tpr)
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    #     plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    #     plt.title(f'{model_name} ROC Curve')
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.legend(loc='lower right')
    #     plt.show()

# 10. Save the best model and vectorizer (Random Forest in this case)
best_model = models['Logistic Regression']
with open('model/sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('model/vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Best model and vectorizer saved successfully.")