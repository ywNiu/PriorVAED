from vae import VAE
import torch
import pandas as pd
from pandas.core.frame import DataFrame
import scanpy as sc
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz
import os

def dataloaderutl(batch_size, num_workers, scdata_dir, pathway_dir, complex_dir, TF_dir, kegg_dir, reactom_dir, place_holder):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

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
    #gene_adj_mtx[gene_adj_mtx > 1] = 9 # 并集加权
    #gene_idx = pd.read_csv('./3priors_fileterd_inuse/pathway_KEGGgraph_adjmatrix_used_gene_idex.csv')
    #gene_idx = gene_idx.iloc[:,0].tolist()
    # 添加n个全连接单元
    ############################################
    n = place_holder
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
    print("mtx.shape = ", mtx_mask.shape)
    dataSel = oriData.X.T
    cat = dataSel.T.toarray() # CELL * GENE
    cat = torch.Tensor(cat)
    lbls = torch.linspace(cat.shape[0],1,cat.shape[0])
    cat_Tdata = torch.utils.data.TensorDataset(cat, lbls)
    train_loader = torch.utils.data.DataLoader(dataset=cat_Tdata, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=cat_Tdata, batch_size=batch_size, shuffle=False)

    classes = ()
    return test_loader, train_loader, classes, mtx_mask, gene_adj_mtx
def save_file(dataset_name,ckpt_save_dir,batch_size, num_workers, scdata_dir, pathway_dir, complex_dir, TF_dir, kegg_dir, reactom_dir, place_holder):
    mnist_test, mnist_train, classes, mtx, gene_adj_mtx = dataloaderutl(batch_size, num_workers, scdata_dir, pathway_dir, complex_dir, TF_dir, kegg_dir, reactom_dir, place_holder)
    model_b = VAE(mask_mtx=mtx,gene_adj_mtx=gene_adj_mtx)
    model_b.load_state_dict(torch.load(os.path.join(ckpt_save_dir, 'model_best.pth'))['state_dict'])
    print(model_b)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(device)
    ensem2d_var = []
    ensem2d_mu = []
    for batch_index, (x, _) in enumerate(mnist_train):
        #x = x.view(x.shape[0],1,1,x.shape[1])
        #print(model_b.encode(x)[1].shape)
        #print(batch_index)
        for item in model_b.encode(x)[1]:
            ensem2d_var.append(item)
        for item in model_b.encode(x)[0]:
            ensem2d_mu.append(item)
    ensem2d_reparameter = []
    for i in range(len(ensem2d_mu)):
        ensem2d_reparameter.append(model_b.reparameterization(ensem2d_mu[i], ensem2d_var[i]))
    print('cell number is: ', len(ensem2d_mu))
    ensem2d_reparameter = [x.tolist() for x in ensem2d_reparameter]
    ensem2d_reparameter = pd.DataFrame(ensem2d_reparameter)
    #sparse_matrix = csr_matrix(ensem2d_reparameter)
    #save_npz('./reparameter.npz', sparse_matrix)
    ensem2d_reparameter.to_csv("./%s/reparameter.csv"%dataset_name,index=False)
    ensem2d_var = [x.tolist() for x in ensem2d_var]
    ensem2d_mu = [x.tolist() for x in ensem2d_mu]
    ensem2d_var = pd.DataFrame(ensem2d_var)
    ensem2d_mu = pd.DataFrame(ensem2d_mu)
    #sparse_matrix = csr_matrix(ensem2d_mu)
    #save_npz('./mu.npz', sparse_matrix)
    #sparse_matrix = csr_matrix(ensem2d_var)
    #save_npz('./var.npz', sparse_matrix)
    ensem2d_var.to_csv("./%s/var.csv"%dataset_name,index=False)
    ensem2d_mu.to_csv("./%s/mu.csv"%dataset_name,index=False)
    print("VAE write done!")
