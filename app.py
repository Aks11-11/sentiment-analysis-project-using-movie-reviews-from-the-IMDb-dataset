from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('model/sentiment_model.h5')

with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        sequences = tokenizer.texts_to_sequences([review])
        padded_sequences = pad_sequences(sequences, padding='post', maxlen=200)
        prediction = model.predict(padded_sequences)
        sentiment = 'Positive' if prediction > 0.5 else 'Negative'
        return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')

if __name__ == '__main__':
    app.run(debug=True)

