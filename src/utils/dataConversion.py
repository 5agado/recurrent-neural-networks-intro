import numpy as np
import itertools
import nltk
import timeit

import utils.constants as const

def tokenize_text(text_lines):
    """
    Split text into sentences, append start and end tokens to each and tokenize
    :param text_lines: list of text lines or list of length one containing all text
    :return: list of sentences
    """
    sentences = itertools.chain(*[nltk.sent_tokenize(line.lower()) for line in text_lines])
    sentences = ["{} {} {}".format(const.SENT_START_TOKEN, x, const.SENT_END_TOKEN) for x in sentences]
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    return tokenized_sentences

def get_words_mappings(tokenized_sentences, vocabulary_size):
    frequence = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = frequence.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(const.UNKNOWN_TOKEN)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    return index_to_word, word_to_index

def get_chars_mappings(text):
    index_to_char = list(set(text))
    char_to_index = dict([(char,i) for i,char in enumerate(index_to_char)])
    return index_to_char, char_to_index

def replace_unknown_words_in(tokenized_sentences, word_to_index):
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else const.UNKNOWN_TOKEN for w in sent]