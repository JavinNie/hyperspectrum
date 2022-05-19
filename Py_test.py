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
from Py_train import *
# 加载模型
net = torch.load('decoder_net.pth')
# result_show(net,test_features,test_labels)

dataFile = 'coded_sig.mat'
data = scio.loadmat(dataFile)
# 生成数据集
coded_sig = torch.from_numpy(np.array(data['B']).T).to(torch.float32)
y_pred = net(coded_sig)
decoded_sig=y_pred.detach().numpy()
outmat = {"decoded_sig": decoded_sig}
scio.savemat("decoded_sig.mat", outmat)

# ._modules.items()