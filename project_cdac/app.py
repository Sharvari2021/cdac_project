from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import json

app = Flask(__name__)

# Load the trained model
model = load_model("your_model.h5")

# Load tokenizer word index from Numpy array
tokenizer_word_index = np.load("tokenizer_word_index.npy", allow_pickle=True).item()

# Create a new tokenizer and set its word index
loaded_tokenizer = Tokenizer()
loaded_tokenizer.word_index = tokenizer_word_index

sequence_length = 5  # Adjust this based on your model

# Dictionary to map word indices to words
word_index_dict = {index: word for word, index in loaded_tokenizer.word_index.items()}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form["input_text"]
        input_sequence = loaded_tokenizer.texts_to_sequences([input_text])[0]
        input_sequence = pad_sequences([input_sequence], maxlen=sequence_length, padding="pre")
        predicted_probs = model.predict(input_sequence)[0]
        predicted_word_id = np.argmax(predicted_probs)
        
        # Get the predicted word from the dictionary
        predicted_word = word_index_dict.get(predicted_word_id, "<OOV>")
        
        print("Predicted Probabilities:", predicted_probs)
        print("Predicted Word ID:", predicted_word_id)
        print("Predicted Word:", predicted_word)
        
        return render_template("index.html", input_text=input_text, predicted_word=predicted_word)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
