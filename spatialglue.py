import random

import pandas
from pandas import get_dummies
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
import squidpy as sq
import pandas as pd
import scanpy as sc
from sklearn.neighbors import kneighbors_graph

from SpatialGlue.preprocess import clr_normalize_each_cell, pca, fix_seed, construct_neighbor_graph
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
from SpatialGlue.utils import clustering

def moranI_score(adata, key):
    g = kneighbors_graph(adata.obsm['spatial'], 6, mode='connectivity', metric='euclidean')
    one_hot = get_dummies(adata.obs[key])
    moranI = sc.metrics.morans_i(g, one_hot.values.T).mean()
    return moranI

# load necessary datasets including spatial transcriptome, spatial proteome and scRNA-seq
dataset = 'human_lymph_node_rep1'  # this is the only data with human annotation
adata_omics1 = sc.read_h5ad('Dataset/' + dataset + '/adata_RNA.h5ad')
adata_omics1.var_names_make_unique()
adata_omics2 = sc.read_h5ad('Dataset/' + dataset + '/adata_ADT.h5ad')
adata_omics2.var_names_make_unique()
# Specify data type
data_type = '10x'

# RNA
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)

# Protein
adata_omics2.X = adata_omics2.X.toarray()
adata_omics2 = clr_normalize_each_cell(adata_omics2)
sc.pp.scale(adata_omics2)
adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)

data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type, n_neighbors=8)

# %% train model
ari = []
moranI = []
for _ in range(10):
    model = Train_SpatialGlue(data, datatype=data_type, random_seed=random.randint(0, 10000), epochs=200)
    output = model.train()

    # %% evaluation
    adata = adata_omics1.copy()
    adata.obsm['feat1'] = adata_omics1.obsm['feat']
    adata.obsm['feat2'] = adata_omics2.obsm['feat']
    key = 'SpatialGlue'
    # key = 'emb_latent_omics2'
    adata.obsm[key] = output[key].copy()
    tool = 'mclust'  # mclust, leiden, and louvain
    clustering(adata, key=key, add_key=key, n_clusters=6, method=tool, use_pca=True)

    # visualization
    import matplotlib.pyplot as plt
    fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
    sc.pp.neighbors(adata, use_rep='SpatialGlue', n_neighbors=10)
    sc.tl.umap(adata)

    sc.pl.umap(adata, color='SpatialGlue', ax=ax_list[0], title='SpatialGlue', s=20, show=False)
    sc.pl.embedding(adata, basis='spatial', color='SpatialGlue', ax=ax_list[1], title='SpatialGlue', s=25, show=False)

    plt.tight_layout(w_pad=0.3)
    plt.show()

    prediction = adata.obs[key]

    ari.append(adjusted_rand_score(adata.obs['cluster'], prediction))

    # TODO: verify the computation of the moran's I score
    moranI.append(moranI_score(adata, key))

fig, ax = plt.subplots()
ax.boxplot(adata.uns["moranI"]["I"])
plt.show()

sc.pp.neighbors(adata, use_rep='spatial')
adata.obsm['prediction'] = adata.obs['SpatialGlue'].to_numpy()
print('Moran\'s I score from scanpy is: ' + str(sc.metrics.morans_i(adata, obsm='prediction')))
sc.pp.neighbors(adata, use_rep='SpatialGlue', key_added='SpatialGlue', n_neighbors=51)
sc.pp.neighbors(adata, use_rep='feat1', key_added='feat1', n_neighbors=51)
sc.pp.neighbors(adata, use_rep='feat2', key_added='feat2', n_neighbors=51)
print('Jaccard similarity score is: ' + str(((adata.obsp['SpatialGlue_distances'].toarray() * adata.obsp[
    'feat1_distances'].toarray() > 0).sum(1) / (adata.obsp['SpatialGlue_distances'].toarray() + adata.obsp[
    'feat1_distances'].toarray() > 0).sum(1)).mean()) + '+' + str(((adata.obsp['SpatialGlue_distances'].toarray() *
                                                                    adata.obsp['feat2_distances'].toarray() > 0).sum(
    1) / (adata.obsp['SpatialGlue_distances'].toarray() + adata.obsp['feat2_distances'].toarray() > 0).sum(1)).mean()))
