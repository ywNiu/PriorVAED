from torch import nn
import torch
import torch.nn.functional as F
from customized_linear import CustomizedLinear

class VAE(nn.Module):

    def __init__(self, mask_mtx, gene_adj_mtx, input_dim=3000, h_dim=3000, h1_dim=3000, z_dim=100):
        # 调用父类方法初始化模块的state
        super(VAE, self).__init__()
        self.mask_mtx = torch.tensor(mask_mtx) # mask_mtx is a numpy.ndarray
        self.input_dim = mask_mtx.shape[0]
        self.h_dim = mask_mtx.shape[1]
        self.h1_dim = h1_dim
        self.z_dim = z_dim
        #self.z_dim = mask_mtx.shape[1]
        self.gene_adj_mtx = gene_adj_mtx
        

        # 编码器 ： [b, input_dim] => [b, z_dim]
        self.fc0 = CustomizedLinear(self.gene_adj_mtx)
        
        #self.fc1 = nn.Linear(self.input_dim, self.h_dim)  # 第一个全连接层
        self.fc1 = CustomizedLinear(self.mask_mtx) # sparse connect, mask_mtx.size()[0] = input_dim; mask_mtx.size()[1] = h_dim
        #self.fc1_1 = nn.Linear(self.h_dim, self.h1_dim)
        
        self.fc2 = nn.Linear(self.h_dim, self.z_dim)  # mu
        self.fc3 = nn.Linear(self.h_dim, self.z_dim)  # log_var
        '''
        self.fc2 = nn.Linear(self.h1_dim, self.z_dim)  # mu
        self.fc3 = nn.Linear(self.h1_dim, self.z_dim)  # log_var
        '''
        # 解码器 ： [b, z_dim] => [b, input_dim]
        self.fc4 = nn.Linear(self.z_dim, self.h_dim)
        #self.fc4 = nn.Linear(self.z_dim, self.h1_dim)
        #self.fc4_1 = nn.Linear(self.h1_dim, self.h_dim)
        #self.fc5 = nn.Linear(self.h_dim, self.input_dim)
        self.fc5 = CustomizedLinear(self.mask_mtx.T)
        self.fc6 = CustomizedLinear(self.gene_adj_mtx.T)

    def forward(self, x):
        """
        向前传播部分, 在model_name(inputs)时自动调用
        :param x: the input of our training model [b, batch_size, 1, 28, 28]
        :return: the result of our training model
        """
        batch_size = x.shape[0]  # 每一批含有的样本的个数
        # flatten  [b, batch_size, 1, 28, 28] => [b, batch_size, 784]
        # tensor.view()方法可以调整tensor的形状，但必须保证调整前后元素总数一致。view不会修改自身的数据，
        # 返回的新tensor与原tensor共享内存，即更改一个，另一个也随之改变。
        x = x.view(batch_size, self.input_dim)  # 一行代表一个样本

        # encoder
        mu, log_var = self.encode(x)
        # reparameterization trick
        sampled_z = self.reparameterization(mu, log_var)
        # decoder
        x_hat = self.decode(sampled_z)
        # reshape
        x_hat = x_hat.view(batch_size, 1, 1, self.input_dim)
        return x_hat, mu, log_var

    def encode(self, x):
        """
        encoding part
        :param x: input image
        :return: mu and log_var
        """
        h0 = F.leaky_relu(self.fc0(x))
        #h = self.fc1(h0)
        h = F.leaky_relu(self.fc1(h0))
        #h1 = F.leaky_relu(self.fc1_1(h))
        log_var = self.fc2(h)
        mu = self.fc3(h)

        return mu, log_var

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  # 这里的“*”是点乘的意思

    def decode(self, z):
        """
        Given a sampled z, decode it back to image
        :param z:
        :return:
        """

        h0 = F.leaky_relu(self.fc4(z))
        #h = F.leaky_relu(self.fc4_1(h0))
        h1 = F.leaky_relu(self.fc5(h0))
        x_hat = F.leaky_relu(self.fc6(h1))
        #x_hat = F.leaky_relu(self.fc5(h1)) 
        return x_hat
