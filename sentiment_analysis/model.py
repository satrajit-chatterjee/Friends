import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvPoolBlock(nn.Module):
    def __init__(self, kernel_size):
        """
        :param kernal_size: tuple of size=2 -> (kernal_height, kernal_width)
                        where width is the filter_layer_depth
        """
        super(ConvPoolBlock, self).__init__()
        self.filter_layer_depth = kernel_size[1]
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=self.filter_layer_depth, kernel_size=kernel_size,
                                bias=self.filter_layer_depth)
        # this max pool layer might need to be modified. I'm not convinced stride is required
        self.pool = nn.MaxPool2d(kernel_size=(1, kernel_size[1] - kernel_size[0] + 1))

    def forward(self, x):
        x = F.relu(self.conv2d(x))
        x = self.pool(x)
        return x


class Net(nn.Module):

    def __init__(self, weights_matrix, num_classes):
        """
        :param num_classes: total number of classes to classify
        :param weights_matrix: weights for the embedding layer,
                        matrix of shape -> (num_embeddings, embedding_dim)
        """
        super(Net, self).__init__()
        self.embedding, _, self.dimension = weights_matrix
        self.conv_pool1 = ConvPoolBlock(kernel_size=(3, self.dimension))
        self.conv_pool2 = ConvPoolBlock(kernel_size=(4, self.dimension))
        self.conv_pool3 = ConvPoolBlock(kernel_size=(5, self.dimension))
        self.dropout = nn.Dropout(p=0.5)
        self.num_classes = num_classes
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.lin_layers = LinearLayers(150*48*4).to(device)  # num_output_channels * H * W
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        x1 = F.relu(self.conv_pool1(x))
        x2 = F.relu(self.conv_pool1(x))
        x3 = F.relu(self.conv_pool1(x))

        x = torch.cat((x1, x2, x3), dim=1).view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.lin_layers(x)
        return x


class LinearLayers(nn.Module):
    def __init__(self, size, num_classes=7, flag=True):
        """
        :param size: size of input features
        :param num_classes: number of classes to be classified into
        :param flag: boolean flag to decide which action to perform
        """
        super(LinearLayers, self).__init__()
        self.fc1 = nn.Linear(in_features=size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)
        self.flag = flag

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.flag is False:
            x = F.relu(self.fc3(x))
        else:
            x = F.log_softmax(self.fc3(x))
        return x