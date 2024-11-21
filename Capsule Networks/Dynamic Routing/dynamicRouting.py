import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torchvision.datasets import CIFAR100, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import TensorDataset
from mainProjects.preCodes.pathIdnt import fullPath, sharedUniq, partWhole, visPClayer, visCClayer, partwholeCC, computeRMA, convLayerPathpathEstimation


KS= 9
nPC, oCh= 8, 32
def trainCapsNet(args, preTrainedModel, channels, ConvChannels, inputSize, classesNum, batch_size):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    
    '''Start loading the model'''
    
    class ConvLayer(nn.Module):
        def __init__(self, in_channels=ConvChannels, out_channels=256):
            '''Constructs the ConvLayer with a specified input and output size.
               param in_channels: input depth of an image, default value = 1
               param out_channels: output depth of the convolutional layer, default value = 256
               '''
            super(ConvLayer, self).__init__()

            # defining a convolutional layer of the specified size
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=KS, stride=1, padding=0)

        def forward(self, x):
            '''Defines the feedforward behavior.
               param x: the input to the layer; an input image
               return: a relu-activated, convolutional layer
               '''
            # applying a ReLu activation to the outputs of the conv layer
            features = F.relu(self.conv(x))  # will have dimensions (batch_size, 20, 20, 256)
            return features

    class PrimaryCaps(nn.Module):
        def __init__(self, num_capsules=nPC, in_channels=256, out_channels=oCh):
            '''Constructs a list of convolutional layers to be used in
                   creating capsule output vectors.
               param num_capsules: number of capsules to create
               param in_channels: input depth of features, default value = 256
               param out_channels: output depth of the convolutional layers, default value = 32
               '''
            super(PrimaryCaps, self).__init__()

            # creating a list of convolutional layers for each capsule I want to create
            # all capsules have a conv layer with the same parameters
            self.capsules = nn.ModuleList([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=KS, stride=2, padding=0)
                for _ in range(num_capsules)])

        def forward(self, x):
            '''Defines the feedforward behavior.
               param x: the input; features from a convolutional layer
               return: a set of normalized, capsule output vectors
               '''
            batch_size = x.size(0)
            u = [capsule(x).view(batch_size, oCh * channels * channels, 1) for capsule in self.capsules]
            u = torch.cat(u, dim=-1)
            u_squash = self.squash(u)
            return u_squash

        def squash(self, input_tensor):
            '''Squashes an input Tensor so it has a magnitude between 0-1.
               param input_tensor: a stack of capsule inputs, s_j
               return: a stack of normalized, capsule output vectors, v_j
               '''
            squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
            scale = squared_norm / (1 + squared_norm)  # normalization coeff
            output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
            return output_tensor

    def dynamic_routing(b_ij, u_hat, squash, routing_iterations=3):

        for iteration in range(routing_iterations):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)
            v_j = squash(s_j)
            if iteration < routing_iterations - 1:
                a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
                b_ij = b_ij + a_ij
        return v_j, a_ij

    class DigitCaps(nn.Module):
        def __init__(self, num_capsules=classesNum, previous_layer_nodes=oCh * channels * channels,
                     in_channels=nPC, out_channels=16):
            super(DigitCaps, self).__init__()

            # setting class variables
            self.num_capsules = num_capsules
            self.previous_layer_nodes = previous_layer_nodes  # vector input (dim=1152)
            self.in_channels = in_channels  # previous layer's number of capsules

            self.W = nn.Parameter(torch.randn(num_capsules, previous_layer_nodes,
                                              in_channels, out_channels))

        def forward(self, u):
            '''Defines the feedforward behavior.
               param u: the input; vectors from the previous PrimaryCaps layer
               return: a set of normalized, capsule output vectors
               '''

            u = u[None, :, :, None, :]
            W = self.W[:, None, :, :, :]
            u_hat = torch.matmul(u, W)

            b_ij = torch.zeros(*u_hat.size())

            # moving b_ij to GPU, if available
            if args.cuda:
                b_ij = b_ij.cuda()

            # update coupling coefficients and calculate v_j
            v_j, c_ij = dynamic_routing(b_ij, u_hat, self.squash, routing_iterations=3)

            return v_j, c_ij, u_hat  # return final vector outputs

        def squash(self, input_tensor):
            '''Squashes an input Tensor so it has a magnitude between 0-1.
               param input_tensor: a stack of capsule inputs, s_j
               return: a stack of normalized, capsule output vectors, v_j
               '''
            # same squash function as before
            squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
            scale = squared_norm / (1 + squared_norm)  # normalization coeff
            output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
            return output_tensor

    class Decoder(nn.Module):
        def __init__(self, input_vector_length=16, input_capsules=classesNum, hidden_dim=512):
            '''Constructs an series of linear layers + activations.
               param input_vector_length: dimension of input capsule vector, default value = 16
               param input_capsules: number of capsules in previous layer, default value = 10
               param hidden_dim: dimensions of hidden layers, default value = 512
               '''
            super(Decoder, self).__init__()

            # calculate input_dim
            input_dim = input_vector_length * input_capsules

            # define linear layers + activations
            self.linear_layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),  # first hidden layer
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim * 2),  # second, twice as deep
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim * 2, ConvChannels * inputSize * inputSize),  # can be reshaped into 28*28 image
                nn.Sigmoid()  # sigmoid activation to get output pixel values in a range from 0-1
            )

        def forward(self, x):
            '''Defines the feedforward behavior.
               param x: the input; vectors from the previous DigitCaps layer
               return: two things, reconstructed images and the class scores, y
               '''

            classes = (x ** 2).sum(dim=-1) ** 0.5
            classes = F.softmax(classes, dim=-1)
            _, max_length_indices = classes.max(dim=1)

            sparse_matrix = torch.eye(classesNum)  # 10 is the number of classes
            if args.cuda:
                sparse_matrix = sparse_matrix.cuda()
            y = sparse_matrix.index_select(dim=0, index=max_length_indices.data)
            # create reconstructed pixels
            x = x * y[:, :, None]
            # flatten image into a vector shape (batch_size, vector_dim)
            flattened_x = x.contiguous().view(x.size(0), -1)
            reconstructions = self.linear_layers(flattened_x)

            return reconstructions, y, classes

    class CapsuleNetwork(nn.Module):
        def __init__(self):
            '''Constructs a complete Capsule Network.'''
            super(CapsuleNetwork, self).__init__()
            self.conv_layer = ConvLayer()
            self.primary_capsules = PrimaryCaps()
            self.digit_capsules = DigitCaps()
            self.decoder = Decoder()

        def forward(self, images):
            '''Defines the feedforward behavior.
               param images: the original MNIST image input data
               return: output of DigitCaps layer, reconstructed images, class scores
               '''
            x = self.conv_layer(images)
            primary_caps_output = self.primary_capsules(x)
            caps_output, routes, u_hat = self.digit_capsules(primary_caps_output)
            caps_output= caps_output.squeeze().transpose(0,1)
            caps_output= caps_output.squeeze()
            caps_output = caps_output.squeeze().transpose(0, 1)

            if (caps_output.shape == torch.Size([10, 16])):
                caps_output= caps_output.unsqueeze(dim=0)
                reconstructions, y, logits = self.decoder(caps_output)
            else:
                caps_output = caps_output.permute(1, 0, 2)
                reconstructions, y, logits = self.decoder(caps_output)

            return caps_output, reconstructions, y, routes, u_hat, logits

    # instantiate and print net
    capsule_net = CapsuleNetwork()
    capsule_net = capsule_net.to(device)
    return capsule_net
