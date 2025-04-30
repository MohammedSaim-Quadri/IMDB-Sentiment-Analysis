import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# Load the IMDB dataset word index
word_index = imdb.get_word_index()
index_from = 3
word_index = {k: (v + index_from) for k, v in word_index.items() if v < 10000}

# Add special tokens
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# Load the pre-trained model
model = load_model('imdb_rnn_model.h5')

def decode_review(encoded_review):
    # decode the review text
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


def preproceess_review(review):
    # preprocess the review text
    words = review.lower().split()
    encoded = [word_index.get(w, 2) for w in words]  # 2 = <UNK>
    padded = sequence.pad_sequences([encoded], maxlen=500)
    return padded


## streamlit app
import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive/negative).")

#user input
user_input = st.text_area("Enter your review here:")
if st.button("Predict"):
    preprocess_input = preproceess_review(user_input)
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]:.4f}")
else:
    st.write("Please enter a review and click on 'Predict' to see the sentiment.")
