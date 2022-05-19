# date:2022-05-17
# Author:Nie Jiewen
# Func:MLP
# fit 32/500 to 500

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



# 加载生成数据集
def Dataset_Gen(N_samples):
    # 加载滤波器
    dataFile = 'FilterHX.mat'
    data = scio.loadmat(dataFile)
    Filter = np.array(data['A']).T
    n=Filter.shape[0] #行数目，即每个滤波器的采样点数
    m=Filter.shape[1] #列数目，即滤波器的数目

    # 生成测试波形数据集
    # 输入数据:
    paraset = []
    labelset = []
    x=(np.arange(0,m)-m/2)*4/m
    for ii in range(N_samples):
        # sig 1
        u=np.random.rand()*x.max()
        d=np.random.rand()/2
        sig = Gaussian(x, u, d)
        # sig 2
        u=np.random.rand()*x.max()
        d=np.random.rand()/2
        sig = sig+Gaussian(x, u, d)
        # norm
        sig = sig * 1/(sig.max()+1e-10)
        coded_sig=np.dot(Filter,sig)
        paraset.append(coded_sig)
        labelset.append(sig)
        # # 双峰的就行
        # plt.plot(para_set.T)
        # plt.show()
    para_set=np.array(paraset)#每个点的数值不超过1，因此将其除以总点数，避免因为点数多少，影响幅值高低
    label_set = np.array(labelset)
    return para_set,label_set

def Gaussian(x, u, d):
    """
    参数:
    x -- 变量
    u -- 均值
    d -- 标准差

    返回:
    p -- 高斯分布值
    """
    ### 代码开始 ### (≈ 3~5 行代码)
    d_2 = d * d * 2
    zhishu = -(np.square(x - u) / d_2)
    exp = np.exp(zhishu)
    pi = np.pi
    xishu = 1 / (np.sqrt(2 * pi) * d)
    p = xishu * exp
    return p
    ### 代码结束 ###
# 对不同的参量分别归一化

paraset,labelset=Dataset_Gen(5)

# 数据集划分
def train_test_split(data, test_size=0.3, shuffle=False, random_state=None):
    '''
    Split DataFrame into random train and test subsets
    Parameters
    ----------
    data : pandas dataframe, need to split dataset.
    test_size : float
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    shuffle : boolean, optional (default=None)
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    '''
    if shuffle:
        data = reset(data, random_state=random_state)
    train = data[int(len(data) * test_size):]
    test = data[:int(len(data) * test_size)]
    return train, test


##################################模型设置########################################

# 构建网络结构
class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.layer1 = torch.nn.Linear(n_feature, 64)  #
        self.layer2 = torch.nn.Linear(64, 256)  #
        self.layer3 = torch.nn.Linear(256, 512)
        self.layer4 = torch.nn.Linear(512, 256)
        self.layer5 = torch.nn.Linear(256, 64)
        self.layer6 = torch.nn.Linear(64, n_output)
    def forward(self, x):  # 这同时也是 Module 中的 forward 功能
        # x = x.to(torch.float32)
        x1 = self.layer1(x)
        x = torch.relu(x1)  #
        x = self.layer2(x)
        x = torch.relu(x)  #
        x = self.layer3(x)
        x = torch.relu(x)  #
        x = self.layer4(x)
        x = torch.relu(x)  #
        x = self.layer5(x)
        x = torch.relu(x) #
        x = self.layer6(x)
        return x

# 网络初始化配置
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.sparse_(m.weight,sparsity=0.1)
        torch.nn.init.xavier_uniform_(m.weight)  # , sparsity=0.1)#(m.weight)
        torch.nn.init.constant_(m.bias,0)

##################################主函数########################################
if __name__ == "__main__":
    # 设置字体为楷体

    torch.manual_seed(0)
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())#计时起点
    train_log_dir = 'logs/train/' + TIMESTAMP

    # data prepare
    strat = time.perf_counter()
    # 加载数据集。输入输出划分
    N_sample=100000
    features,labels = Dataset_Gen(N_sample)

    # 数据集。训练测试划分
    train_features, test_features = train_test_split(features, test_size=0.1, shuffle=True, random_state=0)
    train_labels, test_labels = train_test_split(labels, test_size=0.1, shuffle=True, random_state=0)

    # 输入数据准备完毕
    # # 生成tensor
    train_features = torch.from_numpy(train_features).to(torch.float32)
    train_labels = torch.from_numpy(train_labels).to(torch.float32)
    test_features = torch.from_numpy(test_features).to(torch.float32)
    test_labels = torch.from_numpy(test_labels).to(torch.float32)
    # 组合成训练测试集,便于加载
    train_set = TensorDataset(train_features, train_labels)
    test_set = TensorDataset(test_features, test_labels)
    # 数据加载,迭代器
    nbatch = 1000
    train_data = DataLoader(dataset=train_set, batch_size=nbatch, shuffle=True)
    test_data = DataLoader(dataset=test_set, batch_size=nbatch, shuffle=True)

    # 定义网络，输入输出节点数
    n_feature = train_features.shape[1]
    n_output = train_labels.shape[1]
    net = Net(n_feature, n_output)
    # 网络权重初始化
    net.apply(weight_init)

    # # 定义优化器, SGD Adam等
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)  # , weight_decay=1e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-3, betas=(0.5, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # # 调用损失函数
    # criterion = My_loss()
    criterion = torch.nn.MSELoss(reduction='mean')

    # # 记录损失
    losses = []  # 记录每次迭代后训练的loss
    eval_losses = []  # 测试的
    # tensorboard定义及调用命令
    writer = SummaryWriter('./log')
    # tensorboard - -logdir =./ log
    best_net=net
    best_eval_loss=1e10
    ibatch = 0
    for i in range(200):
        train_loss = 0
        for tdata, tlabel in train_data:
            # 前向传播
            y_pred = net(tdata)
            # 计算损失函数
            loss = criterion(y_pred, tlabel)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # # 累计单批次误差
            train_loss = train_loss + loss.item()
            ibatch = ibatch + 1

        losses.append(train_loss / len(train_data))
        writer.add_scalar('loss', train_loss / len(train_data), i)

        # 测试集进行测试
        eval_loss = 0
        for edata, elabel in test_data:
            # 前向传播
            y_pred = net(edata)
            # 计算损失函数
            loss = criterion(y_pred, elabel)
            eval_loss = eval_loss + loss.item()
        eval_loss=eval_loss / len(test_data)
        if eval_loss<best_eval_loss:
            best_net=net
            best_eval_loss=eval_loss
        net=best_net#更新最佳模型
        current_lr=optimizer.state_dict()['param_groups'][0]['lr']
        eval_losses.append(eval_loss)
        writer.add_scalar('evalloss', eval_loss, i)
        print('epoch: {}, trainloss: {}, evalloss: {}, current_lr: {}'.format(i, train_loss / len(train_data), eval_loss / len(test_data),current_lr))
        # Draw
        num = np.random.randint(0, edata.shape[0])
        yy = y_pred.detach().numpy()
        el = elabel.detach().numpy()
        plt.clf()
        plt.plot(yy[num, :])
        plt.ion()
        plt.plot(el[num, :])
        plt.legend(['pred', 'ori'])
        plt.pause(0.01)
        plt.ioff()
    #
    # num = 5
    # y_pred = net(edata)
    # yy = y_pred.detach().numpy()
    # el = elabel.detach().numpy()
    # plt.figure()
    # plt.plot(yy[num,:])
    # plt.ion()
    # plt.plot(el[num,:])
    # plt.legend(['pred','ori'])
    # plt.ioff()
    # plt.show()
    # #保存模型
    torch.save(best_net,'decoder_net.pth')
