import itertools
import os
import sys

import pandas
import scanpy as sc
from pandas import DataFrame
from scipy.sparse import issparse

from mmvaeplus.preprocess import clr_normalize_each_cell, ST_preprocess
from mmvaeplus.mmvaeplus import MMVAEPLUS

dataset_list = [n for n in os.listdir('Dataset') if os.path.isdir('Dataset/' + n)]
learning_rate_list = [1e-3, 1e-4]
zs_dim_list = [16, 32]
zp_dim_list = [16, 32]
hidden_dim1_list = [256, 512]
hidden_dim2_list = [256, 512]
weight_kl_list = [10, 20]
heads_list = [3, 1]

pss = list(itertools.product(
    *[dataset_list, learning_rate_list, zs_dim_list, zp_dim_list, hidden_dim1_list, hidden_dim2_list, weight_kl_list,
      heads_list]))

start_idx = int(sys.argv[1])
total_idx = int(sys.argv[2])
result_path = 'results/parameter_search_mmvaeplus_' + str(start_idx) + '.csv'
if not os.path.exists(result_path):
    result = DataFrame(
        columns=["dataset", "learning_rate", "zs_dim", "zp_dim", "hidden_dim1", "hidden_dim2", "weight_kl", "heads",
                 "epoch", "n_cluster", "ari", "mi", "nmi", "ami", "hom", "vme", "ave_score", "moran", "jaccard1",
                 "jaccard2", "jaccard"])
    result.to_csv(result_path, index=False)

for i in range(int(start_idx * len(pss) / total_idx), int((start_idx + 1) * len(pss) / total_idx)):
    result = pandas.read_csv(result_path)

    dataset, learning_rate, zs_dim, zp_dim, hidden_dim1, hidden_dim2, weight_kl, heads = pss[i]
    if result[
        (result['dataset'] == dataset) & (result['learning_rate'] == learning_rate) & (result['zs_dim'] == zs_dim) & (
                result['zp_dim'] == zp_dim) & (result['hidden_dim1'] == hidden_dim1) & (
                result['hidden_dim2'] == hidden_dim2) & (result['weight_kl'] == weight_kl) & (
                result['heads'] == heads)].shape[0] > 0:
        continue
    if 'spleen' in dataset:
        n_cluster_list = [5, 3]
    elif 'lymph_node' in dataset:
        n_cluster_list = [6, 10]
    elif 'breast_cancer' in dataset:
        n_cluster_list = [5]
    elif 'thymus' in dataset:
        n_cluster_list = [8]
    elif 'brain' in dataset:
        n_cluster_list = [18]
    else:
        raise Exception('Data not recognized')
    # load necessary datasets including spatial transcriptome and spatial proteome
    adata = []
    for filename in os.listdir('Dataset/' + dataset):
        adata_element = sc.read_h5ad('Dataset/' + dataset + '/' + filename)
        adata_element.var_names_make_unique()
        sc.pp.filter_genes(adata_element, min_cells=1)
        if issparse(adata_element.X):
            adata_element.X = adata_element.X.toarray()
        if adata_element.n_vars > 3000:
            adata_element = ST_preprocess(adata_element, n_top_genes=3000, n_comps=50)
            adata_element = adata_element[:, adata_element.var.highly_variable]
        else:
            adata_element = clr_normalize_each_cell(adata_element)
            sc.pp.pca(adata_element, n_comps=adata_element.n_vars - 1)
        adata.append(adata_element)
    adata_omics1 = adata[0]
    adata_omics2 = adata[1]
    for _ in range(10):
        model = MMVAEPLUS(adata_omics1, adata_omics2, n_neighbors=20, learning_rate=learning_rate, epochs=500,
                          zs_dim=zs_dim, zp_dim=zp_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2,
                          weight_omics1=1, weight_omics2=1, weight_kl=weight_kl, recon_type='nb', heads=heads)
        # train model
        model.train(result_path=result_path, dataset=dataset, test_mode=True, n_cluster_list=n_cluster_list)
