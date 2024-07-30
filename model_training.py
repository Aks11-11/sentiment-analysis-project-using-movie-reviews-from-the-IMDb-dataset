import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report

def create_model(input_length):
    model = Sequential([
        Embedding(input_dim=10000, output_dim=64, input_length=input_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def load_data():
    X_train = np.load('model/X_train.npy')
    X_test = np.load('model/X_test.npy')
    y_train = np.load('model/y_train.npy')
    y_test = np.load('model/y_test.npy')
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = create_model(X_train.shape[1])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint('model/sentiment_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[es, mc])

    model.load_weights('model/sentiment_model.h5')
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
