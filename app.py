from flask import Flask, render_template, request
import pickle
import numpy as np
import re, sys, os, csv, keras, pickle
from keras import regularizers, initializers, optimizers, callbacks
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Input, Flatten, Concatenate
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.callbacks import ModelCheckpoint
app = Flask(__name__)
MAX_NB_WORDS = 56000 # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 30 # max length of text (words) including padding
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 200 # embedding dimensions for word vectors (word2vec/GloVe)

DLModel = pickle.load(open('model.pkl', 'rb'))
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    text = request.form.get("textfile")
    textarr = []
    textarr.insert(text)
   
    sequences_test = tokenizer.texts_to_sequences(textarr)
    data_int_t = pad_sequences(sequences_test, padding='pre', maxlen=(MAX_SEQUENCE_LENGTH-5))
    data_test = pad_sequences(data_int_t, padding='post', maxlen=(MAX_SEQUENCE_LENGTH))
    y_prob = DLModel.predict(data_test)

    print("pppp")
    prediction_final =0
    for n, prediction in enumerate(y_prob):
        pred = y_prob.argmax(axis=-1)[n]
        if(n==0):
            prediction_final = pred
        
   

    
    return render_template('index.html', prediction = prediction_final)

if __name__ == '__main__':
    app.run(port=3000, debug=True)

