import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

nltk.download('punkt')
nltk.download('stopwords')


def load_data():
    df = pd.read_csv(
        'https://raw.githubusercontent.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k/master/train.csv')
    return df['review'], df['sentiment']


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)


def preprocess_data(reviews, sentiments):
    reviews = reviews.apply(preprocess_text)
    le = LabelEncoder()
    sentiments = le.fit_transform(sentiments)
    return reviews, sentiments, le


def tokenize_and_pad(reviews):
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=200)
    return padded_sequences, tokenizer


def save_tokenizer(tokenizer, filename='model/tokenizer.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(tokenizer, f)


def main():
    reviews, sentiments = load_data()
    reviews, sentiments, le = preprocess_data(reviews, sentiments)
    padded_sequences, tokenizer = tokenize_and_pad(reviews)
    save_tokenizer(tokenizer)

    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, sentiments, test_size=0.2, random_state=42)
    np.save('model/X_train.npy', X_train)
    np.save('model/X_test.npy', X_test)
    np.save('model/y_train.npy', y_train)
    np.save('model/y_test.npy', y_test)


if __name__ == '__main__':
    main()
