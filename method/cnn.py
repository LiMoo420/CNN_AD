'''
# This section is used to define the convolutional neural network model 
# and the fully connected multilayer perceptors connected behind the convolutional neural network 
# for four classification tasks: no dementia, very mild dementia, mild dementia, and severe dementia.

# Author: Bo Yin[MC36455*] & Zihan Xue[MC36588*]
# Contact: mc36455@um.edu.mo For Mr.Bo Yin
#          mc36588@um.edu.mo For Ms.Zihan Xue
'''

#==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader
from torchvision import datasets, transforms 
import datetime
#np.random.seed(420)

#==============================================================================

class CustomCNN(nn.Module):
    def __init__(self, 
                 input_channels = 1, num_classes = 4, layers = 4, 
                 kernel_n = 32, kernel_s = 3, pooling_size = 2,
                 activation = 'relu', neurons = 512, dropout = 0.2,
                 pool_stride = 2):
        super(CustomCNN, self).__init__()
        self.layers = nn.ModuleList()
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh

        for i in range(layers):
            self.layers.append(nn.Conv2d(
                in_channels = input_channels if i == 0 else kernel_n * 2 ** (i-1),
                out_channels = kernel_n * 2 ** i,
                kernel_size = kernel_s,
                padding = kernel_s // 2
            ))
            self.layers.append(nn.MaxPool2d(
                kernel_size = pooling_size, stride = pool_stride
            ))
        
        image_size = 128
        # 经过四次池化，图像大小变为128 / 16
        final_image_size = image_size // (2 ** layers)
        # 最后一个卷积层的输出通道数为kernel_n * 2^(layers - 1)
        final_channels = kernel_n * (2 ** (layers - 1))
        # flatten_size为最终特征图的宽度*高度*通道数
        flatten_size = final_channels * (final_image_size ** 2)

        self.flatten_size = flatten_size
        self.fc1 = nn.Linear(self.flatten_size, neurons)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(neurons, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        return x

#==============================================================================

def train_model(model:nn.Module, train_loader, val_loader, epochs, learning_rate, device):
    """
    # The function of training the model.
    #
    # Parameters:
    # =====
    # model: The PyTorch model to train
    # train_loader: DataLoader for training data
    # val_loader: DataLoader used to validate data
    # epochs: Number of training cycles
    # learning_rate: Learning rate
    # device: Computing device ('cuda' or 'cpu')
    """
    model.train()       # Set the model to training mode
    criterion = nn.BCEWithLogitsLoss()           # Define the cross entropy loss function: Binary cross entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)            # Define the optimizer
    
    for epoch in range(epochs):         # Iterate each cycle
        for inputs, targets in train_loader:            # Iterate over each batch of training data
            inputs, targets = inputs.to(device), targets.to(device)         # Move data and labels to computing devices
            optimizer.zero_grad()           # Clear previous gradients
            outputs = model(inputs)         # Perform forward propagation calculation output
            loss = criterion(outputs, targets)          # Calculate the loss
            loss.backward()             # Perform backpropagation to calculate the gradient
            optimizer.step()            # Update model parameters according to gradient
            
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if epoch == 0:
            with open('out/val_loss.txt', 'w') as file:
                pass
        with open('out/val_loss.txt', 'a') as file:
            if file.tell() == 0:
                file.write(f'File Creation Time: {current_time}\n')
            file.write(str(loss) + "\n")

        validate_model(model, val_loader, device, epoch)       # Verify after each cycle

#==============================================================================

def validate_model(model:nn.Module, val_loader, device, epoch:int):
    '''
    # Evaluate the model on a validation set
    '''
    model.eval()
    total = 0           #  `total`: count the total number of samples in the validation set
    correct = 0         # `correct`: record the number of samples that are correctly predicted
    with torch.no_grad():           # Disable gradient calculation during verification
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = torch.softmax(outputs, dim = 1)
            predicted_classes = torch.argmax(predicted, dim = 1)        # extract the category with the highest probability as the prediction result
            one_hot_predicted = F.one_hot(predicted_classes, num_classes = predicted.size(1)).to(device)
            correct_predictions = one_hot_predicted == targets
            correct += correct_predictions.all(dim = 1).sum().item()
            total += targets.size(0)
    accuracy = 100 * (correct / total)
    outputtext = f'The {epoch} epoch Validation Accuracy: {accuracy:.2f}%'
    print(outputtext)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if epoch == 0:
        with open('out/val_accuracy.txt', 'w') as file:
            pass
    with open('out/val_accuracy.txt', 'a') as file:
        if file.tell() == 0:
            file.write(f'File Creation Time: {current_time}\n')
        file.write(outputtext + "\n")

#==============================================================================

def predict_model(model:nn.Module, test_loader, device):
    '''
    # Make predictions on the test set
    '''
    model.eval()
    all_outputs=[]
    with torch.no_grad():
        for batch  in test_loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim = 1)
            all_outputs.append(probabilities.cpu().numpy())
    return np.concatenate(all_outputs, axis = 0)
