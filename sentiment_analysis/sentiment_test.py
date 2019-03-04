import pyprind
import numpy as np
import sys
import sentiment_analysis.train as train
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from sentiment_analysis.train import load_dataset

PATH = "./daily_dialog/test/dialogues_test.txt"
LABELS_PATH = "./daily_dialog/test/dialogues_emotion_test.txt"
writer = SummaryWriter()
dataset, max_len = load_dataset()


def preprocess_dataset():
    """
    This function cleans the dataset and performs post-padding to make sentences of the same length
    :return: weights_matrix ---> matrix containing the GloVe vectors of every word in the dataset vocab
    """
    global EMB_DIM
    dataset, max_len = load_dataset()
    print("Preparing testing data...................")
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
        bar.update()
    dataset = np.array(dataset)
    np.save("processed_test_dataset.npy", dataset)
    print("Test data prepared!")  # <--- save weight matrix in a file so that it does need to be created every time


# preprocess_dataset()
net = train.net
checkpoint = torch.load('./last_model_state.pth')
net.load_state_dict(checkpoint['net_state_dict'])
device = train.device
loaded_dataset = np.load("processed_test_dataset.npy")
loaded_labels = open(LABELS_PATH, "r").readlines()
with torch.no_grad():
    for step in range(len(loaded_dataset)):
        input = loaded_dataset[step]
        outputs = net(input)
        _, predicted = torch.argmax(outputs, 1)
        truth = loaded_labels[step]
        if truth == predicted:
            print(step, " Correct")
        else:
            print(step, "Wrong")
