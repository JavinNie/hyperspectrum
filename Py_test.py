import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import scipy.io as scio
from datetime import datetime
from sklearn.utils import shuffle as reset
from scipy.interpolate import interpn
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

# 加载模型
net = torch.load('decoder_net')
# result_show(net,test_features,test_labels)

dataFile = 'coded_sig.mat'
data = scio.loadmat(dataFile)
# 生成数据集
data_test = np.array(data['B'])
y_pred = net(data_test)

outmat = {"decoded_sig": y_pred}
scio.savemat("decoded_sig.mat", outmat)