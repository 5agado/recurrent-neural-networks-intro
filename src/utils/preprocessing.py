import itertools
import collections
import numpy as np

SENT_START_TOKEN = "SENTENCE_START"
SENT_END_TOKEN = "SENTENCE_END"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
PADDING_TOKEN = "PADDING"


def tokenize_text(text_lines):
    """
    Split text into sentences, append start and end tokens to each and tokenize
    :param text_lines: list of text lines or list of length one containing all text
    :return: list of sentences
    """
    sentences = itertools.chain(*[nltk.sent_tokenize(line.lower()) for line in text_lines])
    sentences = ["{} {} {}".format(SENT_START_TOKEN, x, SENT_END_TOKEN) for x in sentences]
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    return tokenized_sentences


def get_words_mappings(tokenized_sentences, vocabulary_size):
    # Using NLTK
    # frequence = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # vocab = frequence.most_common(vocabulary_size)

    # Using basic counter
    counter = collections.Counter(itertools.chain(*tokenized_sentences))
    vocab = counter.most_common(vocabulary_size)
    index_to_word = [x[0] for x in vocab]
    # Add padding for index 0
    index_to_word.insert(0, PADDING_TOKEN)
    # Append unknown token (with index = vocabulary size + 1)
    index_to_word.append(UNKNOWN_TOKEN)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
    return index_to_word, word_to_index


def get_chars_mappings(text):
    index_to_char = list(set(text))
    char_to_index = dict([(char, i) for i, char in enumerate(index_to_char)])
    return index_to_char, char_to_index


def replace_unknown_words_in(tokenized_sentences, word_to_index):
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]


def load_embeddings(filepath, vocabulary=None):
    """
    Load embeddings dictionary from text file, optionally filtering out all words not present in the given dictionary
    :param filepath: location of embeddings file
    :return:
    """
    embeddings = {}
    with open(filepath) as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            if vocabulary and (word not in vocabulary):
                continue
            else:
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
    return embeddings


def get_embeddings_matrix(embeddings, word_to_index, embedding_dim):
    """
    Get matrix of embeddings for the given words
    :param embeddings:
    :param word_to_index:
    :param embedding_dim:
    :return:
    """
    nb_umatched_words = 0
    embeddings_matrix = np.zeros((len(word_to_index), embedding_dim))
    for word, i in word_to_index.items():
        # if vocab word in embeddings set corresponding vector
        # otherwise we leave all zeros
        if word in embeddings:
            embeddings_matrix[i] = embeddings[word]
        else:
            nb_umatched_words += 1
    print(nb_umatched_words)
    return embeddings_matrix
