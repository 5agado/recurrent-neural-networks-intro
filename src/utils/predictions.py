import numpy as np

import utils.constants as const

def sample_word_idx_from(words_probs, word_to_index):
    unknown_token_idx = word_to_index[const.UNKNOWN_TOKEN]
    sampled_index = unknown_token_idx
    #Sample until a known word is found
    while sampled_index == unknown_token_idx:
        samples = np.random.multinomial(1, words_probs[-1])
        sampled_index = np.argmax(samples)
    return sampled_index

def generate_sentence(predict_fun, min_len, word_to_index, index_to_word):
    end_token_idx = word_to_index[const.SENT_END_TOKEN]
    new_sentence = [word_to_index[const.SENT_START_TOKEN]]
    unknown_token_idx = word_to_index[const.UNKNOWN_TOKEN]
    #Repeat until we get an end token
    while not new_sentence[-1] == end_token_idx:
        words_probs = predict_fun(new_sentence)
        sampled_index = sample_word_idx_from(words_probs, word_to_index)
        #samples = np.random.multinomial(1, words_probs[-1])
        #sampled_index = np.argmax(samples)
        new_sentence.append(sampled_index)
        #Skip if sentence is getting too long
        if len(new_sentence) > 100 or sampled_index == unknown_token_idx:
            return None
    #Skip if not enough words
    if len(new_sentence) < min_len:
        return None
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

def generate_char_sentence(predict_fun, seed, sent_len, char_to_index, index_to_char):
    sentence = seed
    pattern = [char_to_index[char] for char in seed]
    for i in range(sent_len):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x/float(len(char_to_index))
        prediction = predict_fun(x)
        #char_index = sample(prediction)
        char_index = np.random.choice(len(prediction), p=prediction)
        #char_index = np.argmax(prediction)
        sentence += index_to_char[char_index]
        pattern.append(char_index)
        pattern = pattern[1:len(pattern)]
    return sentence

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))