import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from cnn import CustomCNN
import os

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
filedir = 'processed_data'          # Directory of whole splited data
splits = ['train', 'test', 'val']           # Different uses of data
classifications = ['non', 'verymild', 'mild', 'moderate']      # Classification of disease degree

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(
    input_channels = 1, num_classes = len(classifications), layers = layers, 
    kernel_n = kernel_num, kernel_s = kernel_size, pooling_size = pooling_size,
    activation = activation, neurons = full_connection_neurons, dropout = dropout)
model.to(device)

model.load_state_dict(torch.load('out/cnn_model.pth', map_location = device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def save_feature_maps(model, image_path, save_path):
    image = Image.open(image_path).convert('L')  # 转换成灰度图
    image = transform(image).unsqueeze(0).to(device)  # 转换成适合模型的格式

    with torch.no_grad():
        first_conv_layer = model.layers[0]
        feature_maps = first_conv_layer(image)

    fig, axes = plt.subplots(8, 4, figsize=(16, 32))  # Adjusted for an 8x4 grid
    for i in range(32):
        ax = axes[i // 4, i % 4]  # Index into the 8x4 grid
        ax.imshow(feature_maps[0, i].cpu().numpy(), cmap='gray')
        ax.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # Save the figure to the specified path
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

image_path = 'raw_data/moderate_demented/moderate_37.jpg'
save_path = 'out/conv1out.png'

# 可视化特征图
save_feature_maps(model, image_path, save_path)