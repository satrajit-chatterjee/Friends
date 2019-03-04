import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pyprind
import sys

from sentiment_analysis import model
from sentiment_analysis.glove_parser import load_processed_glove


"""
Constants
"""
TRAIN_PATH = "./daily_dialog/train/dialogues_train.txt"
EMO_LABELS_TRAIN_PATH = "./daily_dialog/train/dialogues_emotion_train.txt"
ACT_LABELS_TRAIN_PATH = "./daily_dialog/train/dialogues_act_train.txt"

VAL_PATH = "./daily_dialog/validation/dialogues_validation.txt"
EMO_LABELS_VAL_PATH = "./daily_dialog/validation/dialogues_emotion_validation.txt"
ACT_LABELS_VAL_PATH = "./daily_dialog/validation/dialogues_act_validation.txt"

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
    global TRAIN_PATH, VAL_PATH
    max_len = 0
    val_max_len = 0
    dataset = []
    val_dataset = []
    print("Loading dataset...................")
    f = open(TRAIN_PATH, 'r', encoding="utf-8")
    g = open(VAL_PATH, 'r', encoding="utf-8")
    for l in f:
        line = l.split()
        if len(line) > max_len:
            max_len = len(line)
        dataset.append(line)
    for l in g:
        line = l.split()
        if len(line) > val_max_len:
            val_max_len = len(line)
        val_dataset.append(line)

    print("Dataset loaded! Max length of sentence found is ", max_len)
    return dataset, max_len, val_dataset, val_max_len


def preprocess_dataset():
    """
    This function cleans the dataset and performs post-padding to make sentences of the same length
    :return: weights_matrix ---> matrix containing the GloVe vectors of every word in the dataset vocab
    """
    global EMB_DIM
    glove = load_processed_glove()
    dataset, max_len, val_dataset, val_max_len = load_dataset()
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

    for i, line in enumerate(val_dataset):
        for j in line:
            if j == "?" or j == "," or j == "." or j == "!" or j == ";" or j == "'":
                line.remove(j)
    for i, line in enumerate(val_dataset):
        while len(line) != 278:
            line.append("<PAD>")

    for i, line in enumerate(val_dataset):
        for j, word in enumerate(line):
            word = word.lower()

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
    val_dataset = np.array(val_dataset)
    np.savetxt("weights_matrix.txt", weights_matrix, fmt='%d')
    np.save("processed_dataset.npy", dataset)
    np.save("processed_val_dataset.npy", val_dataset)
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
act_net = model.Net(create_emb_layer(), 4).to(device)
net_total_params = sum(p.numel() for p in net.parameters())
print('total params : ', net_total_params)

criterion = nn.NLLLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True)

loaded_dataset = np.load("processed_dataset.npy")
loaded_val_dataset = np.load("processed_val_dataset.npy")
loaded_emo_labels = open(EMO_LABELS_TRAIN_PATH, "r").readlines()
loaded_act_labels = open(ACT_LABELS_TRAIN_PATH, "r").readlines()
loaded_emo_val_labels = open(EMO_LABELS_VAL_PATH, "r").readlines()
loaded_act_val_labels = open(ACT_LABELS_VAL_PATH, "r").readlines()
iterator = 0
epoch = 0
running_loss = 0.0
running_act_loss = 0.0

val_running_loss = 0.0
val_running_act_loss = 0.0

# checkpoint = torch.load('./checkpoint.pth')
# net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# running_loss = checkpoint['loss']
# net.train()


def train():
    global epoch, running_loss, running_act_loss, val_running_loss, val_running_act_loss, device
    train_hist = np.zeros(EPOCHS)
    train_act_hist = np.zeros(EPOCHS)
    val_hist = np.zeros(EPOCHS)
    val_act_hist = np.zeros(EPOCHS)
    print("TRAINING STARTS!!!")
    for epoch in range(epoch, EPOCHS):
        batch = []
        val_batch = []
        labels = []
        val_labels = []
        act_labels = []
        val_act_labels = []
        iterator = 0
        for step in range(int(math.ceil(len(loaded_dataset) / BATCH_SIZE))):
            for k in range(BATCH_SIZE):
                # print("Loading training batch...................")
                batch.append(loaded_dataset[iterator])
                val_batch.append(loaded_val_dataset[iterator])
                loaded_emo_labels[iterator] = loaded_emo_labels[iterator].replace("\n", "")
                loaded_emo_val_labels[iterator] = loaded_emo_val_labels[iterator].replace("\n", "")
                loaded_act_labels[iterator] = loaded_act_labels[iterator].replace("\n", "")
                loaded_act_val_labels[iterator] = loaded_act_val_labels[iterator].replace("\n", "")
                labels.append(loaded_emo_labels[iterator])
                val_labels.append(loaded_emo_val_labels[iterator])
                val_act_labels.append(loaded_act_val_labels[iterator])
                iterator += 1
            batch_indices = []
            val_batch_indices = []
            for l, i in enumerate(batch):
                temp = []
                for k, j in enumerate(i):
                    temp.append(k)
                batch_indices.append(temp)

            for l, i in enumerate(val_batch):
                temp = []
                for k, j in enumerate(i):
                    temp.append(k)
                val_batch_indices.append(temp)

            batch_indices = torch.from_numpy(np.array(batch_indices).astype(int)).type(torch.LongTensor)
            batch_indices = batch_indices.to(device)
            val_batch_indices = torch.from_numpy(np.array(val_batch_indices).astype(int)).type(torch.LongTensor)
            val_batch_indices = val_batch_indices.to(device)
            labels = torch.from_numpy(np.array(labels).astype(int))
            act_labels = torch.from_numpy(np.array(act_labels).astype(int))
            optimizer.zero_grad()
            outputs = net(batch_indices)
            act_outputs = act_net(batch_indices)
            loss = criterion(outputs, labels.to(device, dtype=torch.long))
            act_loss = criterion(act_outputs, act_labels.to(device, dtype=torch.long))
            #####################
            # VALIDATION RUN
            #####################
            net.eval()
            act_net.eval()
            val_pred = net(val_batch_indices)
            val_emo_loss = criterion(val_pred, val_labels.to(device), dtype=torch.long)
            val_act_pred = act_net(val_batch_indices)
            val_act_loss = criterion(val_act_pred, val_act_labels.to(device), dtype=torch.long)
            val_running_loss += val_emo_loss.item()
            val_running_act_loss += val_act_loss.item()
            net.train()
            act_net.train()
            #####################
            loss.backward()
            act_loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_act_loss += act_loss.item()
        if epoch % 100 == 99:
            print('[EPOCH: %d] loss: %.3f act_loss: %.3f' %
                  (epoch + 1, running_loss / 100, running_act_loss / 100))
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
            }, './checkpoint.pth')
        train_hist[epoch] = running_loss/len(loaded_dataset)
        train_act_hist[epoch] = running_act_loss/len(loaded_dataset)
        val_hist[epoch] = val_running_act_loss/len(loaded_dataset)
        val_act_hist[epoch] = val_running_act_loss/len(loaded_dataset)
        running_loss = 0.0
        running_act_loss = 0.0
    print("Finished Training")
    # saving the final model
    torch.save({'net_state_dict': net.state_dict(),
                }, 'last_model_state.pth')

    plt.plot(train_hist, label="Training Emotion Loss", color="blue")
    plt.plot(val_hist, label="Validation Emotion Loss", color="orange")
    plt.legend()
    plt.show()

    plt.plot(train_act_hist, label="Training Act Loss", color="blue")
    plt.plot(val_act_hist, label="Validation Act Loss", color="orange")
    plt.legend()
    plt.show()

# train()
