from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.layers import LSTM
from keras.layers import TimeDistributed

import os

from model.textGenModel import TextGenModel

def load_model_local(model_path, weights_path, index_to_word_path):
    # Load previously saved model
    with open(model_path, 'r') as f:
        model = model_from_json(f.read())
    # Load weights into model
    model.load_weights(weights_path)

    with open(index_to_word_path) as f:
        index_to_word = [l.strip() for l in f]

    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    model = TextGenModel(model, index_to_word, word_to_index)

    return model

def get_basic_LSTM_model(hidden_size, vocabulary_size,
                         loss='categorical_crossentropy', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(None, vocabulary_size), return_sequences=True))
    model.add(TimeDistributed(Dense(vocabulary_size)))
    model.add(Activation('softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

def get_deep_LSTM_model(hidden_size, vocabulary_size,
                         loss='categorical_crossentropy', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(None, vocabulary_size), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocabulary_size)))
    model.add(Activation('softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# Converts sentence from index to word, and tries to fix spacing
def pretty_print_sentence(sentence, index_to_word):
    words = [index_to_word[x].replace("'", "").strip() for x in sentence]
    words = [w if w in ['.', ',', '!', '?', "'", ';', ':', '..', '...'] else ' ' + w for w in
             words]
    return "".join(words).strip()