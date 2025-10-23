import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import argparse
import os
import time

import einops
import numpy as np
import shutil  
import pandas as pd
#import openpyxl
import scanpy as sc
import torchvision
#import cv2
import torch.nn as nn
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, ToTensor


def get_vaeout(dataset_name):
    vae_output_file_path = '%s/mu.csv'%dataset_name
    return vae_output_file_path


def get_shape():
    return (1, 10, 10)

class DDPM(): 

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        alpha_prev = torch.empty_like(alpha_bars)
        alpha_prev[1:] = alpha_bars[0:n_steps - 1]
        alpha_prev[0] = 1
        self.coef1 = torch.sqrt(alphas) * (1 - alpha_prev) / (1 - alpha_bars)
        self.coef2 = torch.sqrt(alpha_prev) * self.betas / (1 - alpha_bars)

    def sample_forward(self, x, t, eps=None):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res

    def sample_backward(self,
                        img_shape,
                        net,
                        device,
                        simple_var=True,
                        clip_x0=True):
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var, clip_x0)
        return x

    def sample_backward_step(self, x_t, t, net, simple_var=True, clip_x0=True): 
                                                                               

        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n,
                                dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (
                    1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        if clip_x0:
            x_0 = (x_t - torch.sqrt(1 - self.alpha_bars[t]) *
                   eps) / torch.sqrt(self.alpha_bars[t])
            x_0 = torch.clip(x_0, -1, 1)
            mean = self.coef1[t] * x_t + self.coef2[t] * x_0
        else:
            mean = (x_t -
                    (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                    eps) / torch.sqrt(self.alphas[t])
        x_t = mean + noise

        return x_t

class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)

    def forward(self, t):
        return self.embedding(t)


class ResidualBlock(nn.Module):

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.actvation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.actvation2 = nn.ReLU()
        if in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1),
                                          nn.BatchNorm2d(out_c))
        else:
            self.shortcut = nn.Identity()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.actvation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(input)
        x = self.actvation2(x)
        return x


class ConvNet(nn.Module):

    def __init__(self,
                 n_steps,
                 intermediate_channels=[10, 20, 40],
                 pe_dim=10,
                 insert_t_to_all_layers=False):
        super().__init__()
        C, H, W = get_shape()  # 1, 28, 28
        self.pe = PositionalEncoding(n_steps, pe_dim)

        self.pe_linears = nn.ModuleList()
        self.all_t = insert_t_to_all_layers
        if not insert_t_to_all_layers:
            self.pe_linears.append(nn.Linear(pe_dim, C))

        self.residual_blocks = nn.ModuleList()
        prev_channel = C
        for channel in intermediate_channels:
            self.residual_blocks.append(ResidualBlock(prev_channel, channel))
            if insert_t_to_all_layers:
                self.pe_linears.append(nn.Linear(pe_dim, prev_channel))
            else:
                self.pe_linears.append(None)
            prev_channel = channel
        self.output_layer = nn.Conv2d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t):
        n = t.shape[0]
        t = self.pe(t)
        for m_x, m_t in zip(self.residual_blocks, self.pe_linears):
            if m_t is not None:
                pe = m_t(t).reshape(n, -1, 1, 1)
                x = x + pe
            x = m_x(x)
        x = self.output_layer(x)
        return x


class UnetBlock(nn.Module):

    def __init__(self, shape, in_c, out_c, residual=False):
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.activation = nn.ReLU()
        self.residual = residual
        if residual:
            if in_c == out_c:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        out = self.ln(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.residual:
            out += self.residual_conv(x)
        out = self.activation(out)
        return out


class UNet(nn.Module):

    def __init__(self,
                 n_steps,
                 channels=[10, 20, 40, 80],
                 pe_dim=10,
                 residual=False) -> None:
        super().__init__()
        C, H, W = get_shape()
        layers = len(channels)
        Hs = [H]
        Ws = [W]
        cH = H
        cW = W
        for _ in range(layers - 1):
            cH //= 2
            cW //= 2
            Hs.append(cH)
            Ws.append(cW)

        self.pe = PositionalEncoding(n_steps, pe_dim)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pe_linears_en = nn.ModuleList()
        self.pe_linears_de = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        prev_channel = C
        for channel, cH, cW in zip(channels[0:-1], Hs[0:-1], Ws[0:-1]):
            self.pe_linears_en.append(
                nn.Sequential(nn.Linear(pe_dim, prev_channel), nn.ReLU(),
                              nn.Linear(prev_channel, prev_channel)))
            self.encoders.append(
                nn.Sequential(
                    UnetBlock((prev_channel, cH, cW),
                              prev_channel,
                              channel,
                              residual=residual),
                    UnetBlock((channel, cH, cW),
                              channel,
                              channel,
                              residual=residual)))
            self.downs.append(nn.Conv2d(channel, channel, 2, 2))
            prev_channel = channel

        self.pe_mid = nn.Linear(pe_dim, prev_channel)
        channel = channels[-1]
        self.mid = nn.Sequential(
            UnetBlock((prev_channel, Hs[-1], Ws[-1]),
                      prev_channel,
                      channel,
                      residual=residual),
            UnetBlock((channel, Hs[-1], Ws[-1]),
                      channel,
                      channel,
                      residual=residual),
        )
        prev_channel = channel
        for channel, cH, cW in zip(channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):
            self.pe_linears_de.append(nn.Linear(pe_dim, prev_channel))
            self.ups.append(nn.ConvTranspose2d(prev_channel, channel, 2, 2))
            self.decoders.append(
                nn.Sequential(
                    UnetBlock((channel * 2, cH, cW),
                              channel * 2,
                              channel,
                              residual=residual),
                    UnetBlock((channel, cH, cW),
                              channel,
                              channel,
                              residual=residual)))

            prev_channel = channel

        self.conv_out = nn.Conv2d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t):
        n = t.shape[0]
        t = self.pe(t)
        encoder_outs = []
        for pe_linear, encoder, down in zip(self.pe_linears_en, self.encoders,
                                            self.downs):
            pe = pe_linear(t).reshape(n, -1, 1, 1)
            x = encoder(x + pe)
            encoder_outs.append(x)
            x = down(x)
        pe = self.pe_mid(t).reshape(n, -1, 1, 1)
        x = self.mid(x + pe)
        for pe_linear, decoder, up, encoder_out in zip(self.pe_linears_de,
                                                       self.decoders, self.ups,
                                                       encoder_outs[::-1]):
            pe = pe_linear(t).reshape(n, -1, 1, 1)
            x = up(x)

            pad_x = encoder_out.shape[2] - x.shape[2]
            pad_y = encoder_out.shape[3] - x.shape[3]
            x = F.pad(x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2,
                          pad_y - pad_y // 2))
            x = torch.cat((encoder_out, x), dim=1)
            x = decoder(x + pe)
        x = self.conv_out(x)
        return x
def sigmoid_normalization(data):
    normalized_data = 1 / (1 + np.exp(-data))
    return normalized_data
def normlization_minus1to1(data):
    normalized_data = data*2-1
    return normalized_data
def dataloader_diff(dataset_name, batch_size=200, num_workers=2): #不补充
    transform = transforms.Compose([
        transforms.ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2)
    ])

    oriData = pd.read_csv(get_vaeout(dataset_name))
    dataSel = oriData.values
    cat = torch.Tensor(dataSel)
    label =  np.arange(dataSel.shape[0])
    label = [int(x) for x in label]
    lbls = torch.tensor(label)
    cat_Tdata = torch.utils.data.TensorDataset(cat, lbls)
    train_loader = torch.utils.data.DataLoader(dataset=cat_Tdata, batch_size=batch_size, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=cat_Tdata, batch_size=batch_size, shuffle=True)

    return test_loader, train_loader


def train(dataset_name, batch_size, n_steps, n_epochs, ddpm: DDPM, net, device='cuda', ckpt_path='/ddpm/model.pth'):
    print('batch size:', batch_size)
    n_steps = n_steps
    dataloader = dataloader_diff(dataset_name, batch_size,2)[0]
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)
    img_shape = get_shape()
    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0

        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.view(current_batch_size, 1, img_shape[1], img_shape[2])
            x = x.to(device)
            t = torch.randint(0, n_steps, (current_batch_size, )).to(device)
            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x, t, eps)
            eps_theta = net(x_t, t.reshape(current_batch_size, 1))
            loss = loss_fn(eps_theta, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(net.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')



convnet_small_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [10, 20],
    'pe_dim': 128
}

convnet_medium_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [10, 10, 20, 20, 40, 40, 80, 80],
    'pe_dim': 256,
    'insert_t_to_all_layers': True
}
convnet_big_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [20, 20, 40, 40, 80, 80, 160, 160],
    'pe_dim': 256,
    'insert_t_to_all_layers': True
}

unet_1_cfg = {'type': 'UNet', 
              'channels': [10, 20, 40, 80], 
              'pe_dim': 128}
unet_res_cfg = {
    'type': 'UNet',
    'channels': [10, 20, 40, 80],
    'pe_dim': 128,
    'residual': True
}


def build_network(config: dict, n_steps):
    network_type = config.pop('type')
    if network_type == 'ConvNet':
        network_cls = ConvNet
    elif network_type == 'UNet':
        network_cls = UNet

    network = network_cls(n_steps, **config)
    return network


def sigmoid_normalization(data):
    normalized_data = 1 / (1 + np.exp(-data))
    return normalized_data
def normlization_minus1to1(data):
    normalized_data = data*2-1
    return normalized_data
def dataloader_diff1(dataset_name, batch_size=100, num_workers=2): #不补充
    transform = transforms.Compose([
        transforms.ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2)
    ])

    oriData = pd.read_csv(get_vaeout(dataset_name))
    dataSel = oriData.values 

    cat = torch.Tensor(dataSel)
    label =  np.arange(dataSel.shape[0])
    label = [int(x) for x in label]
    lbls = torch.tensor(label)
    cat_Tdata = torch.utils.data.TensorDataset(cat, lbls)
    train_loader = torch.utils.data.DataLoader(dataset=cat_Tdata, batch_size=batch_size, shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(dataset=cat_Tdata, batch_size=batch_size, shuffle=False)

    return test_loader, train_loader

configs = [convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg, unet_res_cfg]
def run_diffusion(epoch, steps, batch_size, dataset_name):
    n_epochs = epoch
    n_steps = steps
    config_id = 4
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    batch_size=batch_size
    os.makedirs('./%s/ddpm'%dataset_name, exist_ok=True)
    model_path = './%s/ddpm/model_best.pth'%dataset_name
    
    config = configs[config_id]
    net = build_network(config, n_steps)
    ddpm = DDPM(device, n_steps)
    
    train(dataset_name, batch_size, n_steps, n_epochs, ddpm, net, device=device, ckpt_path=model_path)
    
    net.load_state_dict(torch.load(model_path))
    img_shape = get_shape()
    dataloader = dataloader_diff1(dataset_name,batch_size,2)[0]
    net = net.to(device)
    net = net.eval()
    tsr = []
    with torch.no_grad():
        idx = 0
        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.view(current_batch_size, 1, img_shape[1], img_shape[2])
            x = x.to(device)
            t = torch.tensor(n_steps-1).to(device)
            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x, t, eps)
            for t in range(5 - 1, -1, -1):
                x = ddpm.sample_backward_step(x, t, net, simple_var=True, clip_x0=True)
            tsr.append(x)
            idx += 1
    print('DDPM train Done!')
    res = []
    for i in range(len(tsr)):
        idx = tsr[i]
        for j in range(idx.shape[0]):
            item = idx[j].squeeze().flatten().tolist()
            res.append(item)
    print(len(res))
    res = np.asarray(res)
    res = pd.DataFrame(res)
    res.to_csv('./%s/diffusion_mu.csv'%dataset_name,index=False)
    print('Diffusion write done!')
