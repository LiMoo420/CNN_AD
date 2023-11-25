'''
# This part of the code is used to successfully train the model containing cnn and mlc neural network 

# Author: Bo Yin[MC36455*] & Zihan Xue[MC36588*]
# Contact: mc36455@um.edu.mo For Mr.Bo Yin
#          mc36588@um.edu.mo For Ms.Zihan Xue
'''

#==============================================================================

from method.split import split_folders
from method.cnn import CustomCNN, train_model, predict_model
from method.io import load_data
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
from torchsummary import summary
import datetime
import random
import torch
import numpy as np
import pandas as pd
import os
#np.random.seed(420)

#==============================================================================
# Fuctions
def extract_values_labels(data_list):
    values = [item[0] for item in data_list]
    labels = [item[1] for item in data_list]
    return values, labels

#==============================================================================
# Split filefolders

raw_data_path = 'raw_data'
processed_data_path = 'processed_data'
random_seed = 420
train_ratio, test_ratio, validation_ratio = (0.8, 0.1, 0.1)
split_ratio = [train_ratio, test_ratio, validation_ratio]

if not os.path.exists(processed_data_path):
    split_folders(raw_data_path, processed_data_path, random_seed, split_ratio)         

#==============================================================================
# Hyper parameters

layers = 4
kernel_num = 32
kernel_size = 3
pooling_size = 2
full_connection_neurons = 512
dropout = 0.2
learning_rate = 0.0001
batch_size = 8
epoch = 15
optimizer = 'Adam'
activation = 'relu'
if_plot = True

#==============================================================================
# Load image after splited and Process it

filedir = 'processed_data'          # Directory of whole splited data
splits = ['train', 'test', 'val']           # Different uses of data
classifications = ['non', 'verymild', 'mild', 'moderate']      # Classification of disease degree

train_data, test_data, val_data = load_data(filedir, splits, classifications, transform = True, ifplot = if_plot)     # Load and transform img

random.seed(5)  # 设置随机种子
random.shuffle(test_data)

train_values, train_labels = extract_values_labels(train_data)          # (5119, 128, 128), (5119,)
test_values, test_labels = extract_values_labels(test_data)             # (642, 128, 128), (642,)
val_values, val_labels = extract_values_labels(val_data)                # (639, 128, 128), (639,)

train_images = torch.Tensor(np.array(train_values)).unsqueeze(1)
train_labels_tensor = torch.Tensor(train_labels)
test_images = torch.Tensor(np.array(test_values)).unsqueeze(1)
test_labels_tensor = torch.Tensor(test_labels)
val_images = torch.Tensor(np.array(val_values)).unsqueeze(1)
val_labels_tensor = torch.Tensor(val_labels)

train_dataset = TensorDataset(train_images, train_labels_tensor)
test_dataset = TensorDataset(test_images, test_labels_tensor)
val_dataset = TensorDataset(val_images, val_labels_tensor)

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

'''Define CNN Model and Device'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(
    input_channels = 1, num_classes = len(classifications), layers = layers, 
    kernel_n = kernel_num, kernel_s = kernel_size, pooling_size = pooling_size,
    activation = activation, neurons = full_connection_neurons, dropout = dropout)
model.to(device)

'''Fit and Train model and then predict'''
train_model(model, train_loader, val_loader, epochs = epoch, learning_rate = learning_rate, device = device)
predictions = predict_model(model, test_loader, device)         # shape: (642, 4)

#==============================================================================
# Save results and Analyze

one_hot_predictions = np.zeros_like(predictions)
one_hot_predictions[np.arange(len(predictions)), predictions.argmax(1)] = 1

if not os.path.exists('out'):       # Create out filefolder
    os.makedirs('out')

model_str = str(model)      # Turn model into string
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open('out/cnn_model.txt', 'w') as f:
    if f.tell() == 0:
        f.write(f'File Creation Time: {current_time}\n')
    f.write(model_str)

model_path = 'out/cnn_model.pth'        # save model
torch.save(model.state_dict(), model_path)
np.save('out/prediction_result.npy', one_hot_predictions)
np.save('out/test_labels.npy', np.array(test_labels))

predicted_labels = np.argmax(one_hot_predictions, axis = 1)
true_labels = np.argmax(np.array(test_labels), axis = 1)

accuracy = accuracy_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:')
print(conf_matrix)

with open('out/confusion_matrix.txt', 'w') as f:
    if f.tell() == 0:
        f.write(f'File Creation Time: {current_time}\n')
    f.write(str(conf_matrix))
#np.savetxt('out/confusion_matrix.txt', conf_matrix, fmt = '%d')

print(summary(model, input_size = (1,128,128)))