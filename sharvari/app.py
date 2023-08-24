from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import BertForMaskedLM, BertTokenizer
import torch

app = Flask(__name__)

# Load the LSTM trained model
lstm_model = load_model("lstm_model.h5")

# Load LSTM tokenizer word index from Numpy array
lstm_tokenizer_word_index = np.load("tokenizer_word_index.npy", allow_pickle=True).item()

# Create a new LSTM tokenizer and set its word index
lstm_loaded_tokenizer = Tokenizer()
lstm_loaded_tokenizer.word_index = lstm_tokenizer_word_index

# Load the fine-tuned BERT model
fine_tuned_model = BertForMaskedLM.from_pretrained("fine_tuned_bert_model")
bert_tokenizer = BertTokenizer.from_pretrained("fine_tuned_bert_model")

sequence_length_lstm = 5  # Adjust this based on your LSTM model
sequence_length_bert = 10  # Adjust this based on your BERT model

# Dictionary to map word indices to words for the LSTM model
lstm_word_index_dict = {index: word for word, index in lstm_loaded_tokenizer.word_index.items()}

@app.route("/", methods=["GET", "POST"])
def index():
    lstm_predicted_word = ""
    bert_predicted_word = ""
    
    if request.method == "POST":
        input_text = request.form["input_text"]
        
        # LSTM Model Prediction
        lstm_input_sequence = lstm_loaded_tokenizer.texts_to_sequences([input_text])[0]
        lstm_input_sequence = pad_sequences([lstm_input_sequence], maxlen=sequence_length_lstm, padding="pre")
        lstm_predicted_probs = lstm_model.predict(lstm_input_sequence)[0]
        lstm_predicted_word_id = np.argmax(lstm_predicted_probs)
        lstm_predicted_word = lstm_word_index_dict.get(lstm_predicted_word_id, "<OOV>")
        
        # BERT Model Prediction
        bert_input_ids = bert_tokenizer.encode(input_text, add_special_tokens=True)
        with torch.no_grad():
            bert_outputs = fine_tuned_model(torch.tensor(bert_input_ids).unsqueeze(0))
            bert_predicted_token_id = torch.argmax(bert_outputs.logits[0, -1]).item()
            bert_predicted_word = bert_tokenizer.convert_ids_to_tokens(bert_predicted_token_id)
        
        return render_template(
            "index.html",
            input_text=input_text,
            lstm_predicted_word=lstm_predicted_word,
            bert_predicted_word=bert_predicted_word
        )
    
    return render_template("index.html", lstm_predicted_word=lstm_predicted_word, bert_predicted_word=bert_predicted_word)

if __name__ == "__main__":
    app.run(debug=True)
