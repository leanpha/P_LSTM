# Credit:  Perform sentiment analysis with LSTMs, using TensorFlow - Adit Deshpande @ O*Reilly
# customized by HoangNguyen

#
# IMPORT LIB
# =========================================
import numpy as np
import tensorflow as tf
import collections
# import matplotlib.pyplot as plt
import re

# enable logging
tf.logging.set_verbosity(tf.logging.INFO)
#
# HYPER PARAMETERS
# =========================================
params = {
    'BATCH_SIZE': 128,
    'EPOCHS': 100000,
    'MAX_DOCUMENT_LENGTH': 0,
    'EMBEDDING_SIZE': 128,
    'LSTM_UNITS': 64,
    'NUM_CLASSES': 5
}


#
# LOAD DATA
# =========================================
# load wordsList & wordVectors
def load_npy():
    wordsList = np.load('wordsList.npy')
    wordsList = wordsList.tolist()
    print('wordsList loaded!')
    wordVectors = np.load('wordVectors.npy')
    print('wordVectors loaded!')
    return wordsList, wordVectors


wordsList, wordVectors = load_npy()

print("wordsList's lenght: ", len(wordsList))
print("wordVectors' shape: ", wordVectors.shape)


def clean_lines(f):
    out = True
    for l in f:
        l = l.strip().lower()
        if l:
            if out:
                yield l
                out = False
        else:
            out = True


def clean_string(l):
    l = l[2:]
    l = re.sub("[?\.,\-!%*\(\)\^\$\#\@\"\']", "", l)
    l = l.split()
    return l


def lookup_word_ids(f):
    ids = np.zeros((len(f), params['MAX_DOCUMENT_LENGTH']), dtype='int32')
    line_index = 0
    for l in f:
        word_index = 0
        for w in l:
            try:
                ids[line_index][word_index] = wordsList.index(w)
            except ValueError:
                ids[line_index][word_index] = wordsList.index('UNK')
            word_index += 1
            if word_index >= params['MAX_DOCUMENT_LENGTH']:
                break
        line_index += 1
    return ids


#
# Load train data
# =========================================
with open('train.txt', 'r', encoding='utf-8') as f:
    file = f.read().splitlines()

file = [line for line in clean_lines(file)]

# Extract labels
labels = np.array([int(l[0]) for l in file])

# Extract sentences
sentences = [clean_string(l) for l in file]

# Edit params
params['MAX_DOCUMENT_LENGTH'] = 50
params['EMBEDDING_SIZE'] = wordVectors.shape[1]

# convert sentences to id sequences
ids = lookup_word_ids(sentences)

del file

print("Training data loaded!")

#
# Load test data
# =========================================
with open('test.txt', 'r', encoding='utf-8') as f:
    file = f.read().splitlines()

file = [line for line in clean_lines(file)]
labels_test = np.array([int(l[0]) for l in file])
sentences_test = [clean_string(l) for l in file]
ids_test = lookup_word_ids(sentences_test)
del file

print("Test data loaded!")


#
# CREATE MODEL
# =========================================
def lstm_model_fn(features, labels, mode, params):
    onehot_labels = tf.one_hot(labels, params['NUM_CLASSES'], 1, 0)
    embed = tf.nn.embedding_lookup(wordVectors, features['input'])
#    embed = tf.unstack(embed, axis=1)

    loss, train_op, pred = None, None, None

    lstmCell = tf.contrib.rnn.BasicLSTMCell(params['LSTM_UNITS'])
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
 #   value, _ = tf.nn.static_rnn(lstmCell, embed, dtype=tf.float32)
    value, _ = tf.nn.dynamic_rnn(lstmCell, embed, dtype=tf.float32)
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
 #   last = value[-1]
    logits = tf.layers.dense(last, params['NUM_CLASSES'])
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels))

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.PREDICT:
        pred = tf.nn.softmax(logits=logits)

    eval_metrics_op = {'accuracy': tf.metrics.accuracy(tf.argmax(onehot_labels), tf.argmax(logits))}
    return tf.estimator.EstimatorSpec(mode, pred, loss, train_op, eval_metrics_op)


classifier = tf.estimator.Estimator(model_fn=lstm_model_fn, model_dir='backup_senti_lstm', params=params)

print("LSTM model created!")
#
# CREATE INPUT FUNCTION
# =========================================
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'input': ids},
    labels,
    batch_size=params['BATCH_SIZE'],
    num_epochs=params['EPOCHS'],
    shuffle=True)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'input': ids_test},
    labels_test,
    batch_size=params['BATCH_SIZE'],
    num_epochs=1,
    shuffle=False)

#
# TRAINING
# =========================================
train_spec = tf.estimator.TrainSpec(train_input_fn)
eval_spec = tf.estimator.EvalSpec(test_input_fn, throttle_secs=600)
tf.estimator.train_and_evaluate(classifier, train_spec=train_spec, eval_spec=eval_spec)

print("FINISH TRAINING")
#
# EVALUATE
# =========================================
ev = classifier.evaluate(test_input_fn)
print("FINISH EVALUATING")
print(ev)
