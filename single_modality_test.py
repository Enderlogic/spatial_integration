import numpy as np
from GraphST import GraphST
from anndata import read_h5ad
from GraphST.utils import clustering
from sklearn import metrics
import scanpy as sc

dataset = ['mouse_breast_cancer', 'human_lymph_node', 'mouse_spleen_rep1', 'mouse_spleen_rep2']

for dataname in dataset:
    if 'mouse_spleen' in dataname:
        n_cluster_list = [5, 3]
    elif dataname == 'human_lymph_node':
        n_cluster_list = [6, 10]
    elif dataname == 'mouse_breast_cancer':
        n_cluster_list = [5]
    else:
        raise Exception('Data not recognized')
    adata_srt = read_h5ad('Dataset/' + dataname + '/adata_ADT.h5ad')
    adata_srt.var_names_make_unique()
    model = GraphST.GraphST(adata_srt)
    adata_srt = model.train()
    for nc in n_cluster_list:
        clustering(adata_srt, nc, method='mclust')
        ari = metrics.adjusted_rand_score(adata_srt.obs['domain'], adata_srt.obs['cluster'])
        sc.pl.spatial(adata_srt, color='domain', title='ARI=%.4f'%ari, show=True, spot_size=np.sqrt(2))
    a = 1