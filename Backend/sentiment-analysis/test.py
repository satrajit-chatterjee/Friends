import torch
import torch.nn as nn
import numpy as np
import pyprind
import sys
from sentiment_analysis.glove_parser import load_processed_glove
from sentiment_analysis import model

"Constants"
PATH = "./daily_dialog/train/dialogues_test.txt"
LABELS_PATH = "./daily_dialog/train/dialogues_emotion_test.txt"


def load_dataset():
    """
    This function loads the dataset from file and puts it into a list
    :return: dataset ---> list containing dataset loaded from file
            max_len ---> Length of the longest sentence in the dataset
    """
    global PATH
    max_len = 0
    dataset = []
    print("Loading dataset...................")
    f = open(PATH, 'r', encoding="utf-8")
    for l in f:
        line = l.split()
        if len(line) > max_len:
            max_len = len(line)
        dataset.append(line)
    print("Dataset loaded! Max length of sentence found is ", max_len)
    return dataset, max_len


def preprocess_dataset():
    """
    When deploying modify this function as there should be only one embedding layer that is built on a global vocab
    :return: weights_matrix ---> matrix containing the GloVe vectors of every word in the dataset vocab
    """
    global EMB_DIM
    glove = load_processed_glove()
    dataset, max_len = load_dataset()
    print("Preparing testing data...................")
    matrix_len = len(dataset)
    weights_matrix = np.zeros((matrix_len, EMB_DIM))
    words_found = 0
    bar = pyprind.ProgBar(len(dataset), stream=sys.stdout)

    for i, line in enumerate(dataset):
        for j in line:
            if j == "?" or j == "," or j == "." or j == "!" or j == ";" or j == "'":
                line.remove(j)
    for i, line in enumerate(dataset):
        while len(line) != 278:
            line.append("<PAD>")

    for i, line in enumerate(dataset):
        for j, word in enumerate(line):
            word = word.lower()
            try:
                weights_matrix[j] = glove[word]
                words_found += 1
            except KeyError:
                if word == "<PAD>":
                    weights_matrix[j] = np.zeros((EMB_DIM,))
                else:
                    weights_matrix[j] = np.random.normal(scale=0.6, size=(EMB_DIM,))
        bar.update()
    dataset = np.array(dataset)
    np.savetxt("testing_weights_matrix.txt", weights_matrix, fmt='%d')
    np.save("processed_test_dataset.npy", dataset)
    print("Testing data prepared!")  # <--- save weight matrix in a file so that it does need to be created every time
    return weights_matrix


def create_emb_layer():
    """
    This function creates the embedding layer from the dataset weight_matrix
    :return: emb_layer ---> the embedding layer
            num_embeddings ---> the length of the dataset vocabulary
            embedding_dim ---> the dimension of each embedding
    """
    # weights_matrix = preprocess_dataset()
    weights_matrix = np.loadtxt('testing_weights_matrix.txt', dtype=int)
    num_embeddings = weights_matrix.shape[0]
    embedding_dim = weights_matrix.shape[1]
    weights_matrix = torch.from_numpy(weights_matrix).float()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    # torch.save(weights_matrix, "embed_weights.pth")
    # weights_matrix = torch.load("embed_weights.pth")  # <--- Only uncomment when required
    emb_layer.weight.requires_grad = False  # <--- Might need to be set to true as reqd
    emb_layer.weight = nn.Parameter(weights_matrix)
    return emb_layer, num_embeddings, embedding_dim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
net = model.Net(create_emb_layer(), 7).to(device)
checkpoint = torch.load('./last_model_state.pth')
net.load_state_dict(checkpoint['model_state_dict'])

loaded_dataset = np.load("processed_test_dataset.npy")
loaded_labels = open(LABELS_PATH, "r").readlines()

net.eval()
with torch.no_grad():

