import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvPoolBlock(nn.Module):
    def __init__(self, kernal_size):
        """
        :param kernal_size: tuple of size=2 -> (kernal_height, kernal_width)
                        where width is the filter_layer_depth
        """
        super(ConvPoolBlock, self).__init__()
        self.filter_layer_depth = kernal_size[1]
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=self.filter_layer_depth, kernel_size=kernal_size)
        self.pool = nn.MaxPool2d(kernal_size=(1, kernal_size[1] - kernal_size[0] + 1), stride=(1, 1))

    def forward(self, x):
        x = F.relu(self.conv2d(x))
        x = self.pool(x)
        return torch.flatten(x)


class Net(nn.Module):

    def __init__(self, weights, num_classes):
        """
        :param num_classes: total number of classes to classify
        :param weights: weights for the embedding layer,
                        matrix of shape -> (num_embeddings, embedding_dim)
        """
        super(Net, self).__init__()
        self.dimension = weights.shape[1]
        self.embedding = nn.Embedding.from_pretrained(embeddings=weights, freeze=True)
        self.conv_pool1 = ConvPoolBlock(kernal_size=(3, self.dimension))
        self.conv_pool2 = ConvPoolBlock(kernal_size=(4, self.dimension))
        self.conv_pool3 = ConvPoolBlock(kernal_size=(5, self.dimension))
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(in_features=self.dimension * 3, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.embedding(x)

        x1 = self.conv_pool1(x)
        x2 = self.conv_pool1(x)
        x3 = self.conv_pool1(x)

        x = torch.cat((x1, x2, x3))
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))

        return x
