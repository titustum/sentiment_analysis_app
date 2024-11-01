# app.py
from flask import Flask, request, jsonify, render_template
import pickle

# Load the saved model and vectorizer
with open('model/sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    review_text = data['text']
    
    # Preprocess the input text and make prediction
    text_vectorized = vectorizer.transform([review_text])
    prediction = model.predict(text_vectorized)
    
    return jsonify({'sentiment': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, port=5599)
