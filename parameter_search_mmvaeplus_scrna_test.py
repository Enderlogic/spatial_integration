import itertools
import os
import sys

import pandas as pd
from pandas import DataFrame
import scanpy as sc

from mmvaeplus_scrna_test.mmvaeplus import MMVAEPLUS

# dataset = ['mouse_spleen_rep2', 'human_lymph_node', 'mouse_spleen_rep1', 'mouse_breast_cancer']
dataname = 'human_lymph_node'
method = 'mmvaeplus'
# load necessary datasets including spatial transcriptome and spatial proteome
adata_omics1 = sc.read_h5ad('Dataset/' + dataname + '/adata_RNA.h5ad')
adata_omics1.var_names_make_unique()
adata_omics2 = sc.read_h5ad('Dataset/' + dataname + '/adata_ADT.h5ad')
adata_omics2.var_names_make_unique()
sc.pp.filter_genes(adata_omics1, min_cells=1)

# proportions = pd.read_csv("Dataset/" + dataname + "/proportions.csv")
# proportions.index = adata_omics1.obs_names
adata_hln = sc.read_h5ad('Dataset/human_lymph_node/adata_hln.h5ad')
proportions = adata_hln.obs.iloc[:, 4:-1].div(adata_hln.obs.iloc[:, 4:-1].sum(axis=1), axis=0)
adata_omics1.obs = pd.concat([adata_omics1.obs, proportions], axis=1)
if 'mouse_spleen' in dataname:
    n_cluster_list = [5, 3]
elif dataname == 'human_lymph_node':
    n_cluster_list = [6, 10]
elif dataname == 'mouse_breast_cancer':
    n_cluster_list = [5]
else:
    raise Exception('Data not recognized')

learning_rate_list = [1e-3, 1e-4]
zs_dim_list = [16, 32]
zp_dim_list = [16, 32]
hidden_dim1_list = [128, 256]
hidden_dim2_list = [128, 256]
weight_omics2_list = [1, 5]
weight_kl_list = [10, 20]
weight_clas_list = [100, 1000]
recon_type_omics1_list = ['zinb', 'nb']
heads_list = [1, 3]
pss = list(itertools.product(
    *[learning_rate_list, zs_dim_list, zp_dim_list, hidden_dim1_list, hidden_dim2_list, weight_omics2_list,
      weight_kl_list, weight_clas_list, recon_type_omics1_list, heads_list]))

start_idx = int(sys.argv[1])
total_idx = int(sys.argv[2])

result_path = 'results/parameter_search_mmvae_test_' + str(start_idx) + '.csv'
if not os.path.exists(result_path):
    result = DataFrame(
        columns=["learning_rate", "zs_dim", "zp_dim", "hidden_dim1", "hidden_dim2", "weight_omics2", "weight_kl",
                 "weight_clas", "recon_type", "heads", "epoch", "n_clusters", "ari", "mi", "nmi", "ami", "hom", "vme",
                 "ave_score"])
    result.to_csv(result_path, index=False)
for i in range(int(start_idx * len(pss) / total_idx), int((start_idx + 1) * len(pss) / total_idx)):
    learning_rate, zs_dim, zp_dim, hidden_dim1, hidden_dim2, weight_omics2, weight_kl, weight_clas, recon_type_omics1, heads = \
        pss[i]

    model = MMVAEPLUS(adata_omics1, adata_omics2, n_neighbors=20, learning_rate=learning_rate, epochs=200,
                      zs_dim=zs_dim, zp_dim=zp_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, weight_omics1=1,
                      weight_omics2=weight_omics2, weight_kl=weight_kl, weight_clas=weight_clas,
                      recon_type_omics1=recon_type_omics1, recon_type_omics2='nb', heads=heads)
    # train model
    model.train(test_mode=True, n_cluster_list=n_cluster_list, result_path=result_path)
