# app.py
from flask import Flask, request, jsonify, render_template
import pickle

# Load the saved model and vectorizer
with open('model/sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

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


import logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/predict2', methods=['POST'])
def predict2():
    try:
        data = request.get_json(force=True)
        review_text = data['text']
        
        if not review_text:
            return jsonify({'error': 'No text provided'}), 400

        logging.debug(f'Received review text: {review_text}')
        # Preprocess and make prediction
        text_vectorized = vectorizer.transform([review_text])
        prediction = model.predict(text_vectorized)

        logging.debug(f'Predicted sentiment: {prediction[0]}')
        return jsonify({'sentiment': prediction[0]})
    except Exception as e:
        logging.error(f'Error during prediction: {str(e)}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5599)
