import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import issparse
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
import scanpy as sc

from SpatialGlue.preprocess import clr_normalize_each_cell
from SpatialGlue.utils import clustering
from mmvaeplus.preprocess import ST_preprocess
from mmvaeplus.mmvaeplus import MMVAEPLUS

# dataset = ['human_lymph_node_rep1', 'mouse_spleen_rep2', 'mouse_spleen_rep1', 'mouse_breast_cancer']
dataset = ['human_lymph_node_rep1']
method = 'mmvaeplus'
n_cluster = 10

for dataname in dataset:
    # load necessary datasets including spatial transcriptome and spatial proteome
    adata_omics1 = sc.read_h5ad('Dataset/' + dataname + '/adata_RNA.h5ad')
    adata_omics1.var_names_make_unique()
    adata_omics2 = sc.read_h5ad('Dataset/' + dataname + '/adata_ADT.h5ad')
    adata_omics2.var_names_make_unique()
    adata_omics1_best = sc.read_h5ad('results/mmvaeplus_human_lymph_node_best.h5ad')
    sc.pp.filter_genes(adata_omics1, min_cells=1)

    # preprocss ST data
    adata_omics1 = ST_preprocess(adata_omics1, n_top_genes=3000, n_comps=50)
    adata_omics1 = adata_omics1[:, adata_omics1.var.highly_variable]

    # preprocess Protein data
    if issparse(adata_omics2.X):
        adata_omics2.X = adata_omics2.X.toarray()
    adata_omics2 = clr_normalize_each_cell(adata_omics2)

    clustering(adata_omics1_best, key=method, add_key=method, n_clusters=n_cluster, method='mclust', use_pca=True)
    prediction = adata_omics1_best.obs[method]
    ari = adjusted_rand_score(adata_omics1_best.obs['cluster'], prediction)

    print(dataname + ', n_cluster=' + str(n_cluster) + ', ari=' + str(ari))

    # visualization
    s = 25
    width = 4
    height = 3
    if 'mouse_spleen' in dataname:
        adata_omics1_best.obsm['spatial'] = np.rot90(
            np.rot90(np.rot90(np.array(adata_omics1_best.obsm['spatial'])).T).T).T
        adata_omics1_best.obsm['spatial'][:, 1] = -1 * adata_omics1_best.obsm['spatial'][:, 1]
        s = 30
    elif dataname == 'mouse_breast_cancer':
        adata_omics1_best.obsm['spatial'][:, 0] = -1 * adata_omics1_best.obsm['spatial'][:, 0]
        adata_omics1_best.obsm['spatial'][:, 1] = -1 * adata_omics1_best.obsm['spatial'][:, 1]
        height = 5
        s = 90

    figure_title = method + ' (' + str(n_cluster) + ' clusters)\nARI: ' + str(round(ari, 5))
    sc.pl.embedding(adata_omics1_best, basis='spatial', color=['cluster', method], title=['Ground truth', figure_title],
                    s=120, wspace=.4)
a = 1