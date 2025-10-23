import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from vae import VAE
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import numpy as np
import pandas as pd
import openpyxl
import scanpy as sc
import time
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz
from scipy.stats import spearmanr
import sklearn
from sklearn.manifold import trustworthiness
from utils import save_file
from diffusion import run_diffusion
# plt.style.use("ggplot")


# 设置模型运行的设备
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# 设置默认参数
parser = argparse.ArgumentParser(description="Variational Auto-Encoder Example")
parser.add_argument('--dataset_name', type=str, default='data_GSE204684_developing_human_cerebral_cortex', metavar='N', help='dataset name')
args_ini = parser.parse_args()
parser.add_argument('--result_dir', type=str, default='./%s/VAEResult'%args_ini.dataset_name, metavar='DIR', help='output directory')
parser.add_argument('--save_dir_ckpt', type=str, default='./%s/checkPoint'%args_ini.dataset_name, metavar='DIR', help='model saving directory')
parser.add_argument('--scData_dir', type=str, default='./data_hvg5000.h5ad', metavar='DIR', help='scData directory')
parser.add_argument('--pathway_dir', type=str, default='./prior_data_results/pathway_use_allgenes_maskedmatrix_filtered_geneNumberOver5.csv', metavar='DIR', help='pathway directory')
parser.add_argument('--complex_dir', type=str, default='./prior_data_results/complex_use_allgenes_maskedmatrix_filtered_geneNumberOver5.csv', metavar='DIR', help='complex directory')
parser.add_argument('--TF_dir', type=str, default='./prior_data_results/TF_use_allgenes_maskedmatrix_filtered_geneNumberOver5.csv', metavar='DIR', help='TF directory')
parser.add_argument('--adjMatrix_kegg_dir', type=str, default='./prior_data_results/pathway_common_with_KEGGgraph_adj_matrix_Sparse.npz', metavar='DIR', help='adjMatrix directory')
parser.add_argument('--adjMatrix_reactom_dir', type=str, default='./prior_data_results/reactome_gene_interaction_adj_matrix_Sparse.npz', metavar='DIR', help='TF directory')
parser.add_argument('--cell_type_key', type=str, default='cell_type', metavar='N', help='cell type key in scData.obs')
parser.add_argument('--batch_size', type=int, default=2000, metavar='N', help='batch size for training(default: 128)')
parser.add_argument('--vae_epochs', type=int, default=10, metavar='N', help='number of epochs to vae train(default: 80)')
parser.add_argument('--diffusion_epochs', type=int, default=10, metavar='N', help='number of epochs to diffusion train(default: 80)')
parser.add_argument('--diffusion_steps', type=int, default=200, metavar='N', help='number of epochs to diffusion train(default: 200)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed(default: 1)')
parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to latest checkpoint(default: None)')
parser.add_argument('--test_every', type=int, default=5, metavar='N', help='test after every epochs')
parser.add_argument('--num_workers', type=int, default=2, metavar='N', help='the number of workers')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate(default: 0.001)')
parser.add_argument('--z_dim', type=int, default=100, metavar='N', help='the dim of latent variable z(default: 20)')
parser.add_argument('--input_dim', type=int, default=1000, metavar='N', help='input dim')
parser.add_argument('--place_holders', type=int, default=5, metavar='N', help='place holder number(default: 5)')
args = parser.parse_args()
kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}


def dataloader(batch_size=args.batch_size, num_workers=args.num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    oriData = sc.read_h5ad(args.scData_dir)

    mtx_mask_0 = pd.read_csv(args.pathway_dir)
    mtx_mask_0 = mtx_mask_0.values
    print('pathway shape: ', mtx_mask_0.shape)
    mtx_mask_1 = pd.read_csv(args.complex_dir)
    mtx_mask_1 = mtx_mask_1.values
    print('complex shape: ', mtx_mask_1.shape)
    mtx_mask_2 = pd.read_csv(args.TF_dir)
    mtx_mask_2 = mtx_mask_2.values
    print('TF shape: ', mtx_mask_2.shape)
    mtx_mask = np.concatenate((mtx_mask_0,mtx_mask_1,mtx_mask_2),axis=1)

    print('final shape: ', mtx_mask.shape)

    sparse_adjmatrix_loaded = load_npz(args.adjMatrix_kegg_dir) #not symmetric
    gene_adj_mtx = sparse_adjmatrix_loaded.toarray()
    gene_adj_mtx = gene_adj_mtx + gene_adj_mtx.T
    gene_adj_mtx = gene_adj_mtx + np.eye(gene_adj_mtx.shape[0])
    reactom_adjmatrix_loaded = load_npz(args.adjMatrix_reactom_dir)
    reactom_gene_regu = reactom_adjmatrix_loaded.toarray()
    gene_adj_mtx = gene_adj_mtx + reactom_gene_regu
    gene_adj_mtx[gene_adj_mtx > 1] = 9 # 交集加权
    #gene_adj_mtx[gene_adj_mtx > 1] = 9 # 并集加权
    #gene_idx = pd.read_csv('./3priors_fileterd_inuse/pathway_KEGGgraph_adjmatrix_used_gene_idex.csv')
    #gene_idx = gene_idx.iloc[:,0].tolist()
    # 添加n个全连接单元
    ############################################
    n = args.place_holders
    if n != 0:
        vec = np.ones((mtx_mask.shape[0], n),dtype=int)
        '''
        empty_genes = [i for i in range(len(np.sum(mtx_mask,axis=1).tolist())) if np.sum(mtx_mask,axis=1).tolist()[i]==0]
        vec = [0]*mtx_mask.shape[0]
        for x in empty_genes:
            vec[x] = 1
        vec = np.array(vec)
        vec = np.tile(vec,n)
        vec = vec.reshape(n,mtx_mask.shape[0])
        vec = vec.T
        '''
        mtx_mask = np.hstack((mtx_mask, vec))
        #gene_adj_mtx = np.hstack((gene_adj_mtx, vec))
    ############################################
    dataSel = oriData.X.T
    cat = oriData.X.toarray()
    cat = torch.tensor(cat)
    label =  oriData.obs[args.cell_type_key].tolist()
    label = [0 for x in label]
    lbls = torch.tensor(label)
    cat_Tdata = torch.utils.data.TensorDataset(cat, lbls)
    train_loader = torch.utils.data.DataLoader(dataset=cat_Tdata, batch_size=batch_size, shuffle=True)
    cat_test = oriData.X.toarray()
    cat_test = torch.tensor(cat_test)
    label_test =  oriData.obs[args.cell_type_key].tolist()
    label_test = [0 for x in label_test]
    lbls_test = torch.tensor(label_test)
    cat_test_Tdata = torch.utils.data.TensorDataset(cat_test, lbls_test)
    test_loader = torch.utils.data.DataLoader(dataset=cat_test_Tdata, batch_size=batch_size, shuffle=True)

    classes = ()
    return test_loader, train_loader, classes, mtx_mask, gene_adj_mtx

def pdist(a,dim=2, p=2):
    dist_matrix = torch.norm(a[:, None]-a, dim, p)
    return dist_matrix 
def pdists(A, squared = False, eps = 1e-8):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min = 0)
    if squared:
        return res
    else:
        res = res.clamp(min = eps).sqrt()
        return res
def loss_function(x_hat, x, mu, log_var, gene_adj_mtx):
    """
    Calculate the loss. Note that the loss includes two parts.
    :param x_hat:
    :param x:
    :param mu:
    :param log_var:
    :return: total loss, BCE and KLD of our model
    """
    # 1. the reconstruction loss.
    # We regard the MNIST as binary classification
    
    x = x.view(x.shape[0],1,1,x.shape[-1])
    #BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
    BCE = F.mse_loss(x_hat, x, reduction='sum')
    # 2. KL-divergence
    # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
    # here we assume that \Sigma is a diagonal matrix, so as to simplify the computation
    KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var) 
    x_f = x.view(x.shape[0],x.shape[-1])
    graph_adj_loss = trustworthiness(x_f.cpu().detach().numpy() , mu.cpu().detach().numpy() , n_neighbors=30, metric='euclidean')
    graph_adj_loss = 1 - graph_adj_loss
    print('graph_adj_loss is: ',graph_adj_loss)
    '''
    graph_adj_loss = 0
    for item in x_hat:
        item = item.squeeze()
        #item = item[gene_idx]
        item = item.view(1, gene_adj_mtx.shape[0])
        item = item.float().t()
        item_mtx = torch.norm(item[:, None]-item, dim=2, p=2)
       # print('item_mtx.size(): ', item_mtx.size())
       # print('item_mtx: ', item_mtx)
        norm_per_row = torch.norm(item_mtx, p=2, dim=1)
        item_mtx = 1 - item_mtx / norm_per_row[:, None]
        gene_adj_mtx = torch.as_tensor(gene_adj_mtx).to(device)
        graph_adj_loss += torch.norm(item_mtx - gene_adj_mtx, 'fro')
    '''
    loss = BCE + KLD + graph_adj_loss
    #loss = BCE + KLD 
    return loss, BCE, KLD, graph_adj_loss


def save_checkpoint(state, is_best, outdir):
    """
    每训练一定的epochs后， 判断损失函数是否是目前最优的，并保存模型的参数
    :param state: 需要保存的参数，数据类型为dict
    :param is_best: 说明是否为目前最优的
    :param outdir: 保存文件夹
    :return:
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    checkpoint_file = os.path.join(outdir, 'checkpoint.pth')  # join函数创建子文件夹，也就是把第二个参数对应的文件保存在'outdir'里
    best_file = os.path.join(outdir, 'model_best.pth')
    torch.save(state, checkpoint_file)  # 把state保存在checkpoint_file文件夹中
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)


def test(model, optimizer, mnist_test, epoch, best_test_loss, gene_adj_mtx):
    test_avg_loss = 0.0
    with torch.no_grad():  # 这一部分不计算梯度，也就是不放入计算图中去
        '''测试测试集中的数据'''
        # 计算所有batch的损失函数的和
        for test_batch_index, (test_x, _) in enumerate(mnist_test):
            test_x = test_x.to(device)
            # 前向传播
            test_x_hat, test_mu, test_log_var = model(test_x)
            # 损害函数值
            test_loss, test_BCE, test_KLD, test_graphloss = loss_function(test_x_hat, test_x, test_mu, test_log_var, gene_adj_mtx)
            test_avg_loss += test_loss

        # 对和求平均，得到每一样本的平均损失
        test_avg_loss /= len(mnist_test.dataset)


        '''保存目前训练好的模型'''
        # 保存模型
        is_best = test_avg_loss < best_test_loss
        best_test_loss = min(test_avg_loss, best_test_loss)
        save_checkpoint({
            'epoch': epoch,  # 迭代次数
            'best_test_loss': best_test_loss,  # 目前最佳的损失函数值
            'state_dict': model.state_dict(),  # 当前训练过的模型的参数
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_dir_ckpt)

        return best_test_loss


def main():
    # Step 1: 载入数据
   # mnist_test, mnist_train, classes = dataloader(args.batch_size, args.num_worker)
    mnist_test, mnist_train, classes, mtx_mask, gene_adj_mtx= dataloader(args.batch_size, args.num_workers)

    # 查看每一个batch的规模
    x, label = iter(mnist_train).__next__()  # 取出第一批(batch)训练所用的数据集
    print("mask_mtx.size = ", mtx_mask.shape)
    print("pathway_connect_sum = ", sum(mtx_mask))
    print("Max(pathway_connect_sum) = ", max(sum(mtx_mask)))
    print("Min(pathway_connect_sum) = ", min(sum(mtx_mask)))
    print("mask_sum = ", np.sum(mtx_mask))
    print("zero_genes_pathway = ", np.sum(sum(mtx_mask)==0))
    print("zero_pathway_genes = ", np.sum(np.sum(mtx_mask,axis=1)==0))
    # Step 2: 准备工作 : 搭建计算流程
    model = VAE(mask_mtx=mtx_mask, gene_adj_mtx=gene_adj_mtx,z_dim=args.z_dim).to(device)  # 生成VAE模型，并转移到GPU上去
    print('The structure of our model is shown below: \n')
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 生成优化器，需要优化的是model的参数，学习率为0.001

    # Step 3: optionally resume(恢复) from a checkpoint
    start_epoch = 0
    best_test_loss = np.finfo('f').max
    if args.resume:
        if os.path.isfile(args.resume):
            # 载入已经训练过的模型参数与结果
            print('=> loading checkpoint %s' % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s' % args.resume)
        else:
            print('=> no checkpoint found at %s' % args.resume)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Step 4: 开始迭代
    loss_epoch = []
    for epoch in range(start_epoch, args.vae_epochs):

        # 训练模型
        # 每一代都要遍历所有的批次
        loss_batch = []
        for batch_index, (x, _) in enumerate(mnist_train):
            # 前向传播
            #x = x.view(x.shape[0],1,1,x.shape[1])
            x = x.to(device)
            x_hat, mu, log_var = model(x)  # 模型的输出，在这里会自动调用model中的forward函数
            loss, BCE, KLD, graph_loss = loss_function(x_hat, x, mu, log_var, gene_adj_mtx)  # 计算损失值，即目标函数
            
            loss_batch.append(loss.item())  # loss是Tensor类型

            # 后向传播
            optimizer.zero_grad()  # 梯度清零，否则上一步的梯度仍会存在
            loss.backward()  # 后向传播计算梯度，这些梯度会保存在model.parameters里面
            optimizer.step()  # 更新梯度，这一步与上一步主要是根据model.parameters联系起来了

            # print statistics every 10 batch
            if (batch_index + 1) % 10 == 0:
                print('Epoch [{}/{}], Batch [{}/{}] : Total-loss = {:.4f}, BCE-Loss = {:.4f}, KLD-loss = {:.4f}, graph-loss = {:.4f}'
                      .format(epoch + 1, args.vae_epochs, batch_index + 1, len(mnist_train.dataset) // args.batch_size,
                              loss.item() / args.batch_size, BCE.item() / args.batch_size,
                              KLD.item() / args.batch_size, graph_loss.item() / args.batch_size))

        # 把这一个epoch的每一个样本的平均损失存起来
        loss_epoch.append(np.sum(loss_batch) / len(mnist_train.dataset))  # len(mnist_train.dataset)为样本个数

        # 测试模型
        if (epoch + 1) % args.test_every == 0:
            best_test_loss = test(model, optimizer, mnist_test, epoch, best_test_loss, gene_adj_mtx)
    return loss_epoch



if __name__ == '__main__':
    loss_epoch = main()
    # 绘制迭代结果
    print(loss_epoch)
    plt.plot(loss_epoch)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    plt.close()
    os.makedirs('./%s'%args.dataset_name, exist_ok=True)
    save_file(args.dataset_name, args.save_dir_ckpt,args.batch_size,args.num_workers, args.scData_dir,args.pathway_dir,args.complex_dir,args.TF_dir,args.adjMatrix_kegg_dir,args.adjMatrix_reactom_dir,args.place_holders)
    run_diffusion(args.diffusion_epochs, args.diffusion_steps, args.batch_size, args.dataset_name)
