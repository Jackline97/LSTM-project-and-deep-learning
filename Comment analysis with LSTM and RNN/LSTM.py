from gensim.models import KeyedVectors
import gensim
import tensorflow.compat.v1 as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding, Dense, Flatten
from keras.layers import Input, LSTM, Dropout, SimpleRNN
from keras.models import Sequential, Model
from keras import optimizers
from keras.callbacks import LearningRateScheduler
import matplotlib
import matplotlib.pyplot as plt

import os
import numpy as np

from pprint import pprint


def processReviews(paths):
    texts = []
    ratings = []

    for path in paths:
        for file in os.listdir(path):
            # get review
            rating = file.split('_')[1]
            rating = rating.split('.')[0]
            file = os.path.join(path, file)
            with open(file, "r", encoding='utf-8') as f:
                text = []
                for line in f:
                    # do some pre-processing and combine list of words for each review text
                    text += gensim.utils.simple_preprocess(line)
                texts.append(text)
                ratings.append(rating)
    return [texts, ratings]
Xtrain, ytrain = processReviews(["./aclImdb/train/neg/", "./aclImdb/train/pos/"])
Xtest, ytest = processReviews(["./aclImdb/test/neg/", "./aclImdb/test/pos/"])

X = list(Xtrain + Xtest)
y = list(ytrain + ytest)
y = [int(a)>= 7 for a in y]
MAX_SEQUENCE_LENGTH=500

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(y))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
print(labels[0])

embeddings_index = {}
glove_file = './glove.6B/glove.6B.50d.txt'

with open(glove_file, "r", encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM=50
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

    X_train, y_train = data[:25000], labels[:25000]
    X_test, y_test = data[25000:], labels[25000:]
#
def RNN_or_NLTK(Option,state_size, MAX_SEQUENCE_LENGTH, learning_rate, Train_Text, Train_Label, Test_Text, Test_label):
    def accuracy_cal(predictions, labels):
        correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
        accu = (100.0 * correctly_predicted) / predictions.shape[0]
        return accu
    tf.disable_eager_execution()
    Text_1 = tf.placeholder(tf.int64, [None, MAX_SEQUENCE_LENGTH])
    Label_1 = tf.placeholder(tf.float64, [None, 2])
    initial_state = tf.nn.embedding_lookup(embedding_matrix, Text_1)
    Weight = tf.Variable(tf.random_normal_initializer()([state_size, 2]))
    Bias = tf.Variable(tf.random_normal_initializer()([2]))
    if Option == 1:
        run_cell = tf.nn.rnn_cell.BasicRNNCell(state_size, reuse=tf.AUTO_REUSE)
    elif Option == 2:
        run_cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, reuse = tf.AUTO_REUSE)
    outputs, states = tf.nn.dynamic_rnn(run_cell, initial_state, dtype=tf.float64)
    Output_mean = tf.reduce_mean(outputs, axis=1)
    Weight = tf.cast(Weight, tf.float64)
    average_output = tf.cast(Output_mean, tf.float64)
    Bias = tf.cast(Bias, tf.float64)
    prediction = tf.matmul(average_output, Weight) + Bias
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Label_1, logits=prediction)
    loss = tf.reduce_mean(loss)
    learning_rate2 = tf.train.exponential_decay(learning_rate, 50, \
                                                        30, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate2).minimize(loss)
    prediction = tf.nn.softmax(prediction)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Label_1, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float64))

    session = tf.Session()
    with session as sess:
        sess.run(tf.global_variables_initializer())
        # Train network
        batch_size = 500
        epoches = 10
        dropout_prob = 0.4

        Train_epoch = []
        Test_epoch = []

        training_accuracy = []
        training_loss = []
        testing_accuracy = []
        testing_loss = []
        index_list = []
        index = np.arange(len(Train_Text))
        np.random.shuffle(index)
        X_train = Train_Text[index]
        y_train = Train_Label[index]
        iteraiton = len(Train_Text) // batch_size
        Loop_step = 0

        for epoch in range(epoches):
            print('epoch', epoch)

            for step in range(iteraiton):
                index_list.append(Loop_step)
                Loop_step +=1
                batch_x = X_train[step * batch_size: (step + 1) * batch_size]
                batch_y = y_train[step * batch_size: (step + 1) * batch_size]

                sess.run(optimizer, {Text_1: batch_x,
                              Label_1: batch_y})

                # Calculate batch loss and accuracy
                loss1, prediction1= sess.run([loss, prediction], feed_dict={Text_1: batch_x,
                                           Label_1: batch_y})
                acc = accuracy_cal(prediction1, batch_y)
                training_loss.append(loss1)
                training_accuracy.append(acc)
                index = np.arange(len(Test_Text))
                np.random.shuffle(index)
                X_test1 = Test_Text[index]
                y_test1 = Test_label[index]
                x_test_batch = X_test1[:256]
                y_test_batch = y_test1[:256]
                test_loss,prediction2 = sess.run([loss,prediction],{Text_1: x_test_batch,
                                     Label_1: y_test_batch})
                test_acc = accuracy_cal(prediction2, y_test_batch)
                print("Step " + str(step) + ", Minibatch Loss= " + \
                  str(loss1) + ", Training Accuracy= " + \
                  str(acc) + ", Testing Accuracy:", \
                  str(test_acc))
                testing_accuracy.append(test_acc)
                testing_loss.append(test_loss)

                acc_mean = np.mean




    return training_accuracy, testing_accuracy, training_loss, testing_loss, index_list


train_acc, test_acc, train_loss, test_loss, index_list = RNN_or_NLTK(2, 50, MAX_SEQUENCE_LENGTH, 0.1, X_train, y_train, X_test, y_test)
plt.plot(index_list, train_loss, label='batch = 20')

plt.ylabel('loss')
plt.xlabel('steps')
plt.legend()
plt.show()


def vanilla_rnn(num_words, state, lra, dropout, num_outputs=2, emb_dim=50, input_length=500):
    model = Sequential()
    model.add(Embedding(input_dim=num_words + 1, output_dim=emb_dim, input_length=input_length, trainable=False,
                        weights=[embedding_matrix]))
    model.add(SimpleRNN(units=state, input_shape=(num_words, 1), return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(num_outputs, activation='softmax'))

    rmsprop = optimizers.RMSprop(lr=lra)
    model.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

    return model


def lstm_rnn(num_words, state, lra, dropout, num_outputs=2, emb_dim=50, input_length=500):
    model = Sequential()
    model.add(Embedding(input_dim=num_words + 1, output_dim=emb_dim, input_length=input_length, trainable=False,
                        weights=[embedding_matrix]))
    model.add(LSTM(state))
    model.add(Dropout(dropout))
    model.add(Dense(num_outputs, activation='sigmoid'))

    rmsprop = optimizers.RMSprop(lr=lra)
    model.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

    return model


def runModel(state, lr, batch, dropout, model, epoch=5, num_outputs=2, emb_dim=100, input_length=2380):
    num_words = len(word_index)
    if model == "lstm":
        model = lstm_rnn(num_words, state, lr, dropout)
    elif model == "vanilla":
        model = vanilla_rnn(num_words, state, lr, dropout)
    epoch = 10
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch, batch_size=batch, verbose=1)

    testscore = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', testscore[0])
    print('Test accuracy:', testscore[1])

    return [history, testscore]



history, scores = runModel(500, 0.01, 200, 0.5, "lstm")
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.title('model acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['test', 'train'], loc='upper left')
plt.show()
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
print("test accuracy", scores[1])