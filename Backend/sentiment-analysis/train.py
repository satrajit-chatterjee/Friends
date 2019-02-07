import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import pyprind
import sys

from sentiment_analysis import model
from sentiment_analysis.glove_parser import load_processed_glove

writer = SummaryWriter()

"""
Constants
"""
PATH = "./daily_dialog/train/dialogues_train.txt"
LABELS_PATH = "./daily_dialog/train/dialogues_emotion_train.txt"
EMB_DIM = 50
BATCH_SIZE = 64  # Batch Size and number of epochs must be set so that 80,000+ items are iterated in the data set
EPOCHS = 1300  # 64*1300 = 83200
LEARNING_RATE = 0.01


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
    This function cleans the dataset and performs post-padding to make sentences of the same length
    :return: weights_matrix ---> matrix containing the GloVe vectors of every word in the dataset vocab
    """
    global EMB_DIM
    glove = load_processed_glove()
    dataset, max_len = load_dataset()
    print("Preparing training data...................")
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
    np.savetxt("weights_matrix.txt", weights_matrix, fmt='%d')
    np.save("processed_dataset.npy", dataset)
    print("Training data prepared!")  # <--- save weight matrix in a file so that it does need to be created every time
    return weights_matrix


def create_emb_layer():
    """
    This function creates the embedding layer from the dataset weight_matrix
    :return: emb_layer ---> the embedding layer
            num_embeddings ---> the length of the dataset vocabulary
            embedding_dim ---> the dimension of each embedding
    """
    # weights_matrix = preprocess_dataset()
    weights_matrix = np.loadtxt('weights_matrix.txt', dtype=int)
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
print('device selected: ', device)
net = model.Net(create_emb_layer(), 7).to(device)
net_total_params = sum(p.numel() for p in net.parameters())
print('total params : ', net_total_params)

criterion = nn.NLLLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True)

loaded_dataset = np.load("processed_dataset.npy")
loaded_labels = open(LABELS_PATH, "r").readlines()
iterator = 0
epoch = 0
# checkpoint = torch.load('./checkpoint.pth')
# net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# running_loss1 = checkpoint['loss']
# net.train()
print("TRAINING STARTS!!!")
for epoch in range(epoch, EPOCHS):
    batch = []
    labels = []
    running_loss = 0.0
    iterator = 0
    for step in range(math.ceil(len(loaded_dataset)/BATCH_SIZE)):
        for k in range(BATCH_SIZE):
            # print("Loading training batch...................")
            batch.append(loaded_dataset[iterator])
            loaded_labels[iterator] = loaded_labels[iterator].replace("\n", "")
            labels.append(loaded_labels[iterator])
            iterator += 1
        batch_indices = []
        for l, i in enumerate(batch):
            temp = []
            for k, j in enumerate(i):
                temp.append(k)
            batch_indices.append(temp)
        batch_indices = torch.from_numpy(np.array(batch_indices).astype(int)).type(torch.LongTensor)
        batch_indices = batch_indices.to(device)
        labels = torch.from_numpy(np.array(labels).astype(int))
        optimizer.zero_grad()
        outputs = net(batch_indices)
        loss = criterion(outputs, labels.to(device, dtype=torch.long))
        writer.add_scalar('loss', loss, epoch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if epoch % 100 == 99:
        print('[EPOCH: %d] loss: %.3f' %
              (epoch + 1, running_loss / 100))
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
        }, './checkpoint.pth')
print("Finished Training")
# saving the final model
torch.save({'net_state_dict': net.state_dict(),
            }, 'last_model_state.pth')
