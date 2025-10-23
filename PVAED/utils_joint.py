from vae import VAE
import torch
from main import dataloader
import pandas as pd
from pandas.core.frame import DataFrame
import scanpy as sc
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz
import DDPM
from DDPM import DDPM
from DDPM import build_network
import argparse
import time

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print(device)
def get_img_shape():
    return (1, 10, 10)

def dataloaderutl(scdata_dir, fine_data, pathway_dir, complex_dir, TF_dir, kegg_dir, reactom_dir, batch_size, num_workers, place_holder):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    #oriData = pd.read_excel("Allen-RNA-part-1.xlsx", index_col=0)
    oriData = sc.read_h5ad(scdata_dir)
    mtx_mask_0 = pd.read_csv(pathway_dir)
    mtx_mask_0 = mtx_mask_0.values
    print('pathway shape: ', mtx_mask_0.shape)
    mtx_mask_1 = pd.read_csv(complex_dir)
    mtx_mask_1 = mtx_mask_1.values
    print('complex shape: ', mtx_mask_1.shape)
    mtx_mask_2 = pd.read_csv(TF_dir)
    mtx_mask_2 = mtx_mask_2.values
    print('TF shape: ', mtx_mask_2.shape)
    mtx_mask = np.concatenate((mtx_mask_0,mtx_mask_1,mtx_mask_2),axis=1)

    print('final shape: ', mtx_mask.shape)

    sparse_adjmatrix_loaded = load_npz(kegg_dir) #not symmetric
    gene_adj_mtx = sparse_adjmatrix_loaded.toarray()
    gene_adj_mtx = gene_adj_mtx + gene_adj_mtx.T
    gene_adj_mtx = gene_adj_mtx + np.eye(gene_adj_mtx.shape[0])
    reactom_adjmatrix_loaded = load_npz(reactom_dir)
    reactom_gene_regu = reactom_adjmatrix_loaded.toarray()
    gene_adj_mtx = gene_adj_mtx + reactom_gene_regu
    gene_adj_mtx[gene_adj_mtx > 1] = 9 # 交集加权
    # 添加n个全连接单元
    ############################################
    n = place_holder
    if n != 0:
        vec = np.ones((mtx_mask.shape[0], n),dtype=int)
        mtx_mask = np.hstack((mtx_mask, vec))
        #gene_adj_mtx = np.hstack((gene_adj_mtx, vec))
    ############################################
    # fine-tunning data followed
    #####################################################
    fine_Data = sc.read_h5ad(fine_data)
    fine_gene = set(fine_Data.var_names.tolist())
    oriData_gene = set(oriData.var_names.tolist())
    intersect_gene = fine_gene.intersection(oriData_gene)
    gene_idx_ori = []
    gene_idx_fine = []
    for gene in intersect_gene:
        gene_idx_ori.append(oriData.var_names.tolist().index(gene))
        gene_idx_fine.append(fine_Data.var_names.tolist().index(gene))
    fineData_final = np.zeros((fine_Data.shape[0], oriData.shape[1]))
    data = fine_Data.X.toarray()
    for i in range(len(intersect_gene)):
        fineData_final[:, gene_idx_ori[i]] = data[:, gene_idx_fine[i]]
    #oriData_gene_num = oriData.shape[1]
    #fineData_gene_num = fine_Data.shape[1]
    #padding = np.zeros((fineData.shape[0], oriData_gene_num-fineData_gene_num),dtype=float)
    #fineData_final = np.hstack((data, padding))
    #####################################################
    
    
    cat = fineData_final
    cat = torch.Tensor(cat)
    lbls = torch.linspace(cat.shape[0],1,cat.shape[0])
    cat_Tdata = torch.utils.data.TensorDataset(cat, lbls)
    train_loader = torch.utils.data.DataLoader(dataset=cat_Tdata, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=cat_Tdata, batch_size=batch_size, shuffle=False)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    return test_loader, train_loader, classes, mtx_mask, gene_adj_mtx


def ddpm_back(ddpm_after_model_path, train_data, batch_size, n_steps, backward_steps):
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
    configs = [convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg, unet_res_cfg]
    config_id = 4
    config = configs[config_id]
    net = build_network(config, n_steps)
    net.load_state_dict(torch.load('%s'%(ddpm_after_model_path)))
    batch_size=batch_size
    img_shape = get_img_shape()
    #dataloader = dataloader_diff1(batch_size,2)[0]

    net = net.to(device)
    net = net.eval()
    ddpm = DDPM(device, n_steps, backward_steps)
    tsr = []
    dataloader = train_data
    time_s = time.time()
    with torch.no_grad():
        idx = 0
        for x in dataloader:
            current_batch_size = x.shape[0]
            x = x.view(current_batch_size, 1, img_shape[1], img_shape[2])
            x = x.to(device)
            t = torch.tensor(n_steps - 1).to(device)
            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x, t, eps)
            for t in range(backward_steps - 1, -1, -1):
                #x_t = ddpm.sample_backward_step(x_t, t, net, simple_var=True, clip_x0=True)
                x = ddpm.sample_backward_step(x, t, net, simple_var=True, clip_x0=True)
            #tsr.append(x_t)
            tsr.append(x)
            idx += 1
    res = []
    for i in range(len(tsr)):
        idx = tsr[i]
        for j in range(idx.shape[0]):
            item = idx[j].squeeze().flatten().tolist()
            res.append(item)
    res = np.asarray(res)
    res = torch.from_numpy(res)
    res = res.view(batch_size,100)
    res = res.to(device)
    res = res.type(torch.float)
    time_d = time.time()
    print('DDPM sample back used time: %.5f sec'%(time_d - time_s))
    return res

def save_file(dataset_name,ddpm_after_model_path,batch_size, num_workers, scdata_dir, fine_data, pathway_dir, complex_dir, TF_dir, kegg_dir, reactom_dir, place_holder, vae_after_dir, n_steps, sample_back_steps):
    
    mnist_test, mnist_train, classes, mtx,gene_adj_mtx = dataloaderutl(scdata_dir, fine_data, pathway_dir, complex_dir, TF_dir, kegg_dir, reactom_dir, batch_size, num_workers, place_holder)
    model_b = VAE(mask_mtx=mtx, gene_adj_mtx=gene_adj_mtx)
    model_b.load_state_dict(torch.load("%s/model_best.pth"%(vae_after_dir))['state_dict'])
    print(model_b)  
    
    ensem2d_var = []
    ensem2d_mu = []
    ddpm_out_total = []
    for batch_index, (x, _) in enumerate(mnist_train):
        batch_size = x.shape[0]
        #x = x.view(x.shape[0],1,1,x.shape[1])
        #print(model_b.encode(x)[1].shape)
        for item in model_b.encode(x)[1]:
            ensem2d_var.append(item)
        for item in model_b.encode(x)[0]:
            ensem2d_mu.append(item)
        train_loader = torch.utils.data.DataLoader(dataset=model_b.encode(x)[0], batch_size =batch_size, shuffle=False)
        ddpm_out = ddpm_back(ddpm_after_model_path, train_loader, batch_size, n_steps, sample_back_steps)
        print(ddpm_out.size())
        ddpm_out_total.append(ddpm_out)
    ensem2d_reparameter = []
    for i in range(len(ensem2d_mu)):
        ensem2d_reparameter.append(model_b.reparameterization(ensem2d_mu[i], ensem2d_var[i]))
    ensem2d_reparameter = [x.tolist() for x in ensem2d_reparameter]
    ensem2d_reparameter = pd.DataFrame(ensem2d_reparameter)
    ensem2d_reparameter.to_csv("./%s/joint_reparameter.csv"%dataset_name,index=False)
    ensem2d_var = [x.tolist() for x in ensem2d_var]
    ensem2d_mu = [x.tolist() for x in ensem2d_mu]
    ensem2d_var = pd.DataFrame(ensem2d_var)
    ensem2d_mu = pd.DataFrame(ensem2d_mu)
    ensem2d_var.to_csv("./%s/joint_var.csv"%dataset_name,index=False)
    ensem2d_mu.to_csv("./%s/joint_mu.csv"%dataset_name,index=False)
    
    ddpm_out_total = [x.tolist() for x in ddpm_out_total]
    ddpm_out_total_now = []
    for x in ddpm_out_total:
        for y in x:
            ddpm_out_total_now.append(y)
    print('ddpm_out_total cell number: ', len(ddpm_out_total_now))
    ddpm_out_total_now = pd.DataFrame(ddpm_out_total_now)
    ddpm_out_total_now.to_csv("./%s/joint_ddpmout.csv"%dataset_name,index=False)
    print("joint write done!")
