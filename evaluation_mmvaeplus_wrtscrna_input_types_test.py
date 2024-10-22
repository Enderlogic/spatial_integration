import os

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.sparse import issparse
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
import pandas as pd
import scanpy as sc

from SpatialGlue.preprocess import clr_normalize_each_cell
from SpatialGlue.utils import clustering
from mmvaeplus.preprocess import ST_preprocess, pse_srt_from_scrna
from mmvaeplus.mmvaeplus import MMVAEPLUS

dataset = ['mouse_breast_cancer', 'human_lymph_node', 'mouse_spleen_rep1', 'mouse_spleen_rep2']
# dataset = ['mouse_breast_cancer', 'human_lymph_node']
result = DataFrame(columns=['dataset', 'embedding_type', 'metrics', 'result'])
n_repeat = 10
embedding_types = ['z', 'zs', 'z_omics1', 'z_omics2', 'zs_omics1', 'zs_omics2', 'zp_omics1', 'zp_omics2']
for dataname in dataset:
    if dataname == 'human_lymph_node':
        n_cluster = 10
        epochs = 100
    elif dataname == 'mouse_breast_cancer':
        n_cluster = 5
        epochs = 50
    else:
        n_cluster = 3
        epochs = 600
    # load necessary datasets including spatial transcriptome and spatial proteome
    adata_omics1 = sc.read_h5ad('Dataset/' + dataname + '/adata_RNA.h5ad')
    adata_omics1.var_names_make_unique()
    adata_omics2 = sc.read_h5ad('Dataset/' + dataname + '/adata_ADT.h5ad')
    adata_omics2.var_names_make_unique()
    sc.pp.filter_genes(adata_omics1, min_cells=1)

    # preprocss ST data
    adata_omics1 = ST_preprocess(adata_omics1, n_top_genes=3000, n_comps=50)
    adata_omics1 = adata_omics1[:, adata_omics1.var.highly_variable]

    # preprocess Protein data
    if issparse(adata_omics2.X):
        adata_omics2.X = adata_omics2.X.toarray()
    adata_omics2 = clr_normalize_each_cell(adata_omics2)
    sc.pp.pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)

    adata_best = adata_omics1.copy()
    for i in range(n_repeat):
        model = MMVAEPLUS(adata_omics1, adata_omics2, None, n_neighbors=20, learning_rate=1e-3, epochs=epochs,
                          zs_dim=16, zp_dim=16, hidden_dim1=256, hidden_dim2=256, weight_omics1=1, weight_omics2=200,
                          weight_kl=10)
        # train model
        model.train(test_mode=False)
        embedding = model.encode_test()
        # %% evaluation
        adata = adata_omics1.copy()
        for et in embedding_types:
            adata.obsm[et] = embedding[et].copy()
            clustering(adata, key=et, add_key=et, n_clusters=n_cluster, method='mclust', use_pca=False)
            prediction = adata.obs[et]
            ari = adjusted_rand_score(adata_best.obs['cluster'], prediction)
            mi = mutual_info_score(adata_best.obs['cluster'], prediction)
            nmi = normalized_mutual_info_score(adata_best.obs['cluster'], prediction)
            ami = adjusted_mutual_info_score(adata_best.obs['cluster'], prediction)
            hom = homogeneity_score(adata_best.obs['cluster'], prediction)
            vme = v_measure_score(adata_best.obs['cluster'], prediction)
            ave_score = (ari + mi + nmi + ami + hom + vme) / 6

            result.loc[len(result.index)] = [dataname, et, 'ari', ari]
            result.loc[len(result.index)] = [dataname, et, 'mi', mi]
            result.loc[len(result.index)] = [dataname, et, 'nmi', nmi]
            result.loc[len(result.index)] = [dataname, et, 'ami', ami]
            result.loc[len(result.index)] = [dataname, et, 'hom', hom]
            result.loc[len(result.index)] = [dataname, et, 'vme', vme]
            result.loc[len(result.index)] = [dataname, et, 'average', ave_score]
            print(dataname + ' ' + str(i) + 'th iteration, embedding type is: ' + et + ', ari=' + str(ari))

result.to_csv('results/evaluation_embedding_types_test.csv', index=False)
