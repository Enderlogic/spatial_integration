import random

from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
import squidpy as sq
import pandas as pd
import scanpy as sc

from SpatialGlue.preprocess import clr_normalize_each_cell, pca, fix_seed, construct_neighbor_graph
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
from SpatialGlue.utils import clustering


# load necessary datasets including spatial transcriptome, spatial proteome and scRNA-seq
dataset = 'human_lymph_node'  # this is the only data with human annotation
adata_omics1 = sc.read_h5ad('Dataset/' + dataset + '/adata_RNA.h5ad')
adata_omics1.var_names_make_unique()
adata_omics2 = sc.read_h5ad('Dataset/' + dataset + '/adata_ADT.h5ad')
adata_omics2.var_names_make_unique()
ground_truth = pd.read_csv('Dataset/' + dataset + '/annotation.csv') if dataset == 'human_lymph_node' else None
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

data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type)

# %% train model
model = Train_SpatialGlue(data, datatype=data_type, random_seed=random.randint(0, 10000))
output = model.train()

# %% evaluation
adata = adata_omics1.copy()
adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
adata.obsm['SpatialGlue'] = output['SpatialGlue'].copy()
adata.obsm['alpha'] = output['alpha']
adata.obsm['alpha_omics1'] = output['alpha_omics1']
adata.obsm['alpha_omics2'] = output['alpha_omics2']
tool = 'mclust'  # mclust, leiden, and louvain
clustering(adata, key='SpatialGlue', add_key='SpatialGlue', n_clusters=10, method=tool, use_pca=True)

prediction = adata.obs['SpatialGlue']

if ground_truth is not None:
    adata.obs['ground_truth'] = ground_truth['manual-anno'].values
    ari = adjusted_rand_score(prediction, adata.obs['ground_truth'])
    mi = mutual_info_score(prediction, adata.obs['ground_truth'])
    nmi = normalized_mutual_info_score(prediction, adata.obs['ground_truth'])
    ami = adjusted_mutual_info_score(prediction, adata.obs['ground_truth'])
    hom = homogeneity_score(prediction, adata.obs['ground_truth'])
    vme = v_measure_score(prediction, adata.obs['ground_truth'])
    ave_score = (ari + mi + nmi + ami + hom + vme) / 6
    print('Average score is: ' + str(ave_score))

# TODO: verify the computation of the moran's I score
sq.gr.spatial_neighbors(adata)
sq.gr.spatial_autocorr(adata, attr='obs', mode='moran', genes='SpatialGlue')
print('Moran\'s I score is: ' + str(adata.uns["moranI"]['I'][0]))
