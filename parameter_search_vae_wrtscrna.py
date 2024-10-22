import itertools
import os
import sys
import scanpy as sc
from pandas import DataFrame
from scipy.sparse import issparse

from mmvaeplus.preprocess import clr_normalize_each_cell, ST_preprocess
from mmvaeplus.mmvaeplus import MMVAEPLUS

# dataset_list = ['mouse_spleen_rep1', 'mouse_breast_cancer', 'mouse_spleen_rep2', 'human_lymph_node']
dataset_list = ['human_lymph_node']
learning_rate_list = [1e-3, 1e-4]
zs_dim_list = [16, 32]
zp_dim_list = [16, 32]
hidden_dim1_list = [256, 512]
hidden_dim2_list = [256, 512]
weight_omics2_list = [.5, 1, 1.5]
weight_kl_list = [10, 20]
n_neighbors_list = [20, 40]
recon_type_list = ['zinb', 'nb']
heads_list = [3, 1]

pss = list(itertools.product(
    *[dataset_list, learning_rate_list, zs_dim_list, zp_dim_list, hidden_dim1_list, hidden_dim2_list,
      weight_omics2_list, weight_kl_list, n_neighbors_list, recon_type_list, heads_list]))

start_idx = int(sys.argv[1])
total_idx = int(sys.argv[2])
result_path = 'results/parameter_search_vae_' + str(start_idx) + '.csv'
if not os.path.exists(result_path):
    result = DataFrame(
        columns=["dataset", "learning_rate", "zs_dim", "zp_dim", "hidden_dim1", "hidden_dim2", "weight_omics2",
                 "weight_kl", "n_neighbors", "recon_type", "heads", "epoch", "n_cluster", "ari", "mi",
                 "nmi", "ami", "hom", "vme", "ave_score"])
    result.to_csv(result_path, index=False)

for i in range(int(start_idx * len(pss) / total_idx), int((start_idx + 1) * len(pss) / total_idx)):
    dataset, learning_rate, zs_dim, zp_dim, hidden_dim1, hidden_dim2, weight_omics2, weight_kl, n_neighbors, recon_type, heads = \
        pss[i]
    if 'mouse_spleen' in dataset:
        n_cluster_list = [5, 3]
    elif dataset == 'human_lymph_node':
        n_cluster_list = [6, 10]
    elif dataset == 'mouse_breast_cancer':
        n_cluster_list = [5]
    else:
        raise Exception('Data not recognized')
    # load necessary datasets including spatial transcriptome and spatial proteome
    adata_srt = sc.read_h5ad('Dataset/' + dataset + '/adata_RNA.h5ad')
    adata_srt.var_names_make_unique()
    sc.pp.filter_genes(adata_srt, min_cells=1)
    adata_spr = sc.read_h5ad('Dataset/' + dataset + '/adata_ADT.h5ad')
    adata_spr.var_names_make_unique()

    # preprocss ST data
    adata_srt = ST_preprocess(adata_srt, n_top_genes=3000, n_comps=50)
    adata_srt = adata_srt[:, adata_srt.var.highly_variable]

    # preprocess Protein data
    if issparse(adata_spr.X):
        adata_spr.X = adata_spr.X.toarray()
    adata_spr = clr_normalize_each_cell(adata_spr)
    sc.pp.pca(adata_spr, n_comps=adata_spr.n_vars - 1)
    for _ in range(10):
        model = MMVAEPLUS(adata_srt, adata_spr, None, n_neighbors=n_neighbors, learning_rate=learning_rate,
                          epochs=500, zs_dim=zs_dim, zp_dim=zp_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2,
                          weight_omics1=1, weight_omics2=weight_omics2 * adata_srt.n_vars / adata_spr.n_vars,
                          weight_kl=weight_kl, recon_type=recon_type, heads=heads)
        # train model
        model.train(result_path=result_path, dataset=dataset, n_cluster_list=n_cluster_list, test_mode=True)
