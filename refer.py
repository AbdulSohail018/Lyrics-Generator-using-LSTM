from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import io
def create_seq(txt, seq_len, step):
    seq = []
    chars_next= []
    for i in range(0, len(txt) - seq_len, step):
        seq.append(txt[i: i + seq_len])
        chars_next.append(txt[i + seq_len])
    return seq, chars_next
def model_build(seq_len, chars):
    model = Sequential()
    model.add(LSTM(128, input_shape=(seq_len, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimize = RMSprop(lr=0.01)#lr= learning rate=0.01
    model.compile(loss='categorical_crossentropy', optimizer=optimize)
    return model
def sample_test(preds, temp=1.0):
    if temp == 0:
        temp = 1
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
def char_extract(txt):
    return sorted(list(set(txt))) 
def char_index(chars):
    return dict((c, i) for i, c in enumerate(chars)), dict((i, c) for i, c in enumerate(chars))
def corpus_read(path):
    with io.open(path, 'r', encoding='utf8') as f:
        return f.read().lower()
def vector(seq, seq_len, chars,index_char, chars_next):
    X = np.zeros((len(seq), seq_len, len(chars)), dtype=np.bool)
    y = np.zeros((len(seq), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(seq):
        for t, char in enumerate(sentence):
            X[i, t, index_char[char]] = 1
        y[i, index_char[chars_next[i]]] = 1
    return X, y

