import csv
import nltk
import numpy as np
import itertools
import timeit

from RNNNumpy import RNNNumpy
import util.io as mio

vocabulary_size = 1000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

print("Reading conversation and load messages")
messages, senders = mio.parseMessagesFromFile("C:\\Users\\Alex\\Documents\\python_workspace\\conversation-analyzer\\src\\resources\\unittest\\test_plotting.txt")

# Read the data and append SENTENCE_START and SENTENCE_END tokens
# Split full message text into sentences
sentences = itertools.chain(*[nltk.sent_tokenize(m.text.lower()) for m in messages])
# Append SENTENCE_START and SENTENCE_END
sentences = ["{} {} {}".format(sentence_start_token, x, sentence_end_token) for x in sentences]
print("Parsed {} sentences.".format(len(sentences)))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found {} unique words tokens.".format(len(word_freq.items())))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print("Using vocabulary size {}.".format(vocabulary_size))
print("The least frequent word in our vocabulary is '{}' and appeared {} times.".format(vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print("\nExample sentence: '{}'".format(sentences[0]))
print("\nExample sentence after Pre-processing: '{}'".format(tokenized_sentences[0]))

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

np.random.seed(10)
model = RNNNumpy(vocabulary_size)
o, s = model.forward_propagation(X_train[10])
print(o.shape)
print(o)

predictions = model.predict(X_train[10])
print(predictions.shape)
print(predictions)

print("Expected Loss for random predictions: {}".format(np.log(vocabulary_size)))
print("Actual loss: {}".format(model.calculate_loss(X_train[:1000], y_train[:1000])))

grad_check_vocab_size = 100
np.random.seed(10)
model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
model.gradient_check([0,1,2,3], [1,2,3,4])

np.random.seed(10)
model = RNNNumpy(vocabulary_size)
print( timeit.repeat("model.numpy_sdg_step(X_train[10], y_train[10], 0.005)",
                     "from __main__ import model, X_train, y_train", number=1))

np.random.seed(10)
# Train on a small subset of the data to see what happens
model = RNNNumpy(vocabulary_size)
losses = model.train_with_sgd(X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)

num_sentences = 10
senten_min_length = 7

def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print(" ".join(sent))