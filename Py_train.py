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
import matplotlib.animation as animation
import os

# 加载生成数据集
def Dataset_Gen(N_samples,noise_flag):
    # 加载滤波器
    dataFile = 'Filter.mat'
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
        num_peak=np.random.randint(1,8)
        sig=0
        for ii in range(num_peak):
            u = np.random.rand() * (x.max() - x.min()) +x.min()
            d = np.random.rand() / 2
            sig = sig+Gaussian(x, u, d)
        # norm
        sig = sig * 1/(sig.max()+1e-10)
        coded_sig=np.dot(Filter,sig)
        #噪声
        if noise_flag==1:
            coded_sig=wgn(coded_sig, snr=50)

        paraset.append(coded_sig)
        labelset.append(sig)
        # # 双峰的就行
        # plt.plot(para_set.T)
        # plt.show()
    para_set=np.array(paraset)#每个点的数值不超过1，因此将其除以总点数，避免因为点数多少，影响幅值高低

    #norm
    outmat = {"coded_norm_coe": para_set.max()}
    scio.savemat("coded_norm_coe.mat", outmat)
    print(para_set.max())
    para_set=para_set/para_set.max()
    # para_set = para_set / 100

    label_set = np.array(labelset)
    return para_set,label_set
def wgn(x, snr):
    len_x, = x.shape
    Ps = np.sum(np.power(x, 2)) / len_x
    Pn = Ps / (np.power(10, snr / 10))
    noise = np.random.randn(len_x) * np.sqrt(Pn)
    return x + noise
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

# paraset,labelset=Dataset_Gen(5)

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
        self.layer1 = torch.nn.Linear(n_feature, 128)  #
        self.layer2 = torch.nn.Linear(128, 256)  #
        self.layer3 = torch.nn.Linear(256, 512)
        self.layer4 = torch.nn.Linear(512, n_output)
        # self.layer5 = torch.nn.Linear(512, n_output)
        # self.layer6 = torch.nn.Linear(64, n_output)
    def forward(self, x):  # 这同时也是 Module 中的 forward 功能
        # x = x.to(torch.float32)
        x1 = self.layer1(x)
        x = torch.relu(x1)  #
        x = self.layer2(x)
        x = torch.relu(x)  #
        x = self.layer3(x)
        x = torch.relu(x)  #
        x = self.layer4(x)
        # x = torch.relu(x)  #
        # x = self.layer5(x)
        # x = torch.relu(x) #
        # x = self.layer6(x)
        return x

# 网络初始化配置
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.sparse_(m.weight,sparsity=0.1)
        torch.nn.init.xavier_uniform_(m.weight)  # , sparsity=0.1)#(m.weight)
        torch.nn.init.constant_(m.bias,0)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model,optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience/2:
                optimizer.param_groups[0]['lr'] *= 0.2
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return optimizer

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'decoder_net.pth')
        # torch.save(model.state_dict(), path)# 这里会存储迄今最优模型的参数
        torch.save(model, 'decoder_net.pth')
        self.val_loss_min = val_loss

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
    noise_flag=1
    features,labels = Dataset_Gen(N_sample,noise_flag)

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
    best_net=Net(n_feature, n_output)
    # 网络权重初始化
    net.apply(weight_init)

    # # 定义优化器, SGD Adam等
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-1, momentum=0.9)  # , weight_decay=1e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.5, 0.99))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # # 调用损失函数
    # criterion = My_loss()
    criterion = torch.nn.MSELoss(reduction='mean')

    # # 记录损失值
    losses = []  # 记录每次迭代后训练的loss
    eval_losses = []  # 测试的

    # tensorboard定义及调用命令
    TBwriter = SummaryWriter('./log')
    # tensorboard - -logdir =./ log
    best_eval_loss=1e10
    ibatch = 0
    #draw
    draw_flag=0
    if draw_flag == 1:
        fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(1, 1, 1, facecolor='white')
        plt.rcParams['font.size'] = 15
        ims = []  # 将每一帧都存进去
    # 早停止函数
    save_path = ".\\"  # 当前目录下
    early_stopping = EarlyStopping(save_path)

    # 循环训练
    for i in range(300):
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
            # # 累计单批次误差
            train_loss = train_loss + loss.item()
            ibatch = ibatch + 1
        train_loss=train_loss / len(train_data)
        TBwriter.add_scalar('loss', train_loss, i)

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
        #学习率递减
        scheduler.step(eval_loss)#LR需要的
        current_lr=optimizer.state_dict()['param_groups'][0]['lr']
        eval_losses.append(eval_loss)
        TBwriter.add_scalar('evalloss', eval_loss, i)
        print('epoch: {}, trainloss: {}, evalloss: {}, current_lr: {}'.format(i, train_loss, eval_loss,current_lr))
        # update Draw
        if draw_flag==1:
            num = np.random.randint(0, edata.shape[0])
            yy = y_pred.detach().numpy()
            el = elabel.detach().numpy()
            wavelength=np.linspace(1200, 1700, elabel.shape[1])
            # plt.cla()
            plt.clf()
            ax = fig.add_subplot(1, 1, 1, facecolor='white')
            frame1, = ax.plot(wavelength,yy[num, :], color='deepskyblue')
            frame2, = ax.plot(wavelength,el[num, :], color='tomato')
            str = 'epoch:{},eval_loss:{}'.format(i, round(eval_loss, 5))
            frame1.axes.title.set_text(str)
            # ax.set_title(str)
            ax.legend(['pred', 'ori'])
            # plt.title('epoch:{},eval_loss:{}'.format(i,round(eval_loss,5)))
            ims.append([frame1,frame2])
            plt.pause(0.01)
        # 早停止
        optimizer=early_stopping(eval_loss, net,optimizer)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练
    if draw_flag==1:
        ani = animation.ArtistAnimation(fig, ims, interval=1000)  # 生成动画
        # 保存成gif
        ani.save("pendulum.gif", writer='pillow')
    # #保存模型
    torch.save(best_net,'decoder_net.pth')
