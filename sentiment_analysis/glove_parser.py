import pickle
import numpy as np
import bcolz


def process_glove():
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='./6B.50.dat', mode='w')
    with open('./glove.6B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir='./6B.50.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open('./6B.50_words.pkl', 'wb'))
    pickle.dump(word2idx, open('./6B.50_idx.pkl', 'wb'))


def load_processed_glove():
    vectors = bcolz.open('./6B.50.dat')[:]
    words = pickle.load(open('./6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open('./6B.50_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}
    return glove