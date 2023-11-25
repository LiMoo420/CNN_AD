'''
This part is used to optimize hyper parameters in cnn and mlc neural network

# Author: Bo Yin[MC36455*] & Zihan Xue[MC36588*]
# Contact: mc36455@um.edu.mo For Mr.Bo Yin
#          mc36588@um.edu.mo For Ms.Zihan Xue
'''


from ray import tune
from ray.tune.schedulers import ASHAScheduler

'''
确定超参数空间：选择你想要优化的超参数，例如学习率、批次大小、网络层的数量和大小、卷积层大小数量、优化器等等。
选择优化策略：可以是网格搜索、随机搜索或更高级的方法如贝叶斯优化。
定义评价标准：确定如何评估模型性能，通常使用交叉验证或保留一部分数据作为验证集。
执行搜索：运行优化过程，对每组超参数配置训练模型，并记录性能指标。
分析结果：确定哪些超参数配置产生了最好的模型性能。
测试最佳配置：在测试集上评估具有最佳超参数配置的模型。
'''


import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import RandomizedSearchCV
from skorch import NeuralNetClassifier
import numpy as np

# 假设 CustomCNN 是您的自定义PyTorch模型
from cnn import CustomCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 为了使用 RandomizedSearchCV，我们需要将 PyTorch 模型封装为一个 scikit-learn 兼容的估计器
class SkorchWrapper(NeuralNetClassifier):
    def __init__(self, criterion=torch.nn.CrossEntropyLoss, **kwargs):
        super(SkorchWrapper, self).__init__(
            module = CustomCNN,
            criterion = criterion,
            **kwargs
        )

# 定义要优化的超参数空间
param_distributions = {
    'lr': [0.01, 0.001, 0.0001],
    'module__num_classes': [4],  # 如果您的类别数是固定的
    'module__layers': [[2, 2], [3, 3]],
    'module__kernel_n': [16, 32],
    'module__kernel_s': [3, 5],
    'module__pooling_size': [2],
    'module__activation': ['relu', 'tanh'],
    'module__neurons': [64, 128],
    'module__dropout': [0.5, 0.75],
    'max_epochs': [10],  # 可以尝试更多的epochs
    'batch_size': [16, 32]
}

# 将 PyTorch 模型封装为 scikit-learn 兼容的估计器
net = SkorchWrapper(
    module = CustomCNN,
    criterion = torch.nn.CrossEntropyLoss,
    optimizer = torch.optim.Adam,
    iterator_train__shuffle = True,
    device = device  # 'cuda' or 'cpu'
)

# 创建 RandomizedSearchCV 实例
rs = RandomizedSearchCV(
    net,
    param_distributions,
    n_iter=10,  # 运行随机搜索的迭代次数
    scoring='accuracy',
    verbose=2,
    cv=3  # 交叉验证的折数
)

# 准备数据，这里假设 train_values, train_labels 已经准备好
# 这里的 X 和 y 应该是 NumPy 数组，而不是 PyTorch 张量
#X_train = np.array(train_values).astype(np.float32)
#y_train = np.array(train_labels).astype(np.int64)

# 训练模型并执行随机搜索
#rs.fit(X_train, y_train)

# 输出最佳参数
print('Best parameters found:', rs.best_params_)

# 使用最佳参数在全数据集上训练模型
best_model = rs.best_estimator_