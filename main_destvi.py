import pickle
import random
import warnings
import numpy as np
import pandas
import torch
from anndata import read_h5ad
from matplotlib import pyplot as plt
from scvi.model import CondSCVI, DestVI
from sklearn.metrics import adjusted_rand_score

from SpatialGlue.utils import clustering
from starfysh import utils, AA, plot_utils
from starfysh.utils import VisiumArguments
import scanpy as sc
import seaborn as sns

warnings.filterwarnings('ignore')

# load necessary datasets including spatial transcriptome, spatial proteome and scRNA-seq
adata_st = read_h5ad('Dataset/human_lymph_node_rep1/adata_RNA.h5ad')
adata_st.var_names_make_unique()
adata_sc = read_h5ad('Dataset/human_lymph_node_rep1/adata_scrna.h5ad')
adata_sc.obs['celltype'] = adata_sc.obs['Subset']
# to_replace = {}
# for ct in adata_sc_omics1.obs['celltype'].unique():
#     ct_split = ct.split('_')
#     ct_new = '_'.join(ct_split[:-1]) + '_combined'
#     to_replace[ct] = ct_new
# adata_sc_omics1.obs = adata_sc_omics1.obs.replace({'celltype': to_replace})
sc.pp.filter_genes(adata_sc, min_counts=10)
adata_sc.layers['counts'] = adata_sc.X.copy()
sc.pp.highly_variable_genes(adata_sc, n_top_genes=3000, subset=True, layer='counts', flavor='seurat_v3')
sc.pp.normalize_total(adata_sc, target_sum=10e4)
sc.pp.log1p(adata_sc)
adata_sc.raw = adata_sc

adata_st.layers['counts'] = adata_st.X.copy()
sc.pp.normalize_total(adata_st, target_sum=10e4)
sc.pp.log1p(adata_st)
adata_st.raw = adata_st

intersect = np.intersect1d(adata_sc.var_names, adata_st.var_names)
adata_st = adata_st[:, intersect].copy()
adata_sc = adata_sc[:, intersect].copy()

CondSCVI.setup_anndata(adata_sc, layer='counts', labels_key='celltype')
model_sc = CondSCVI(adata_sc, weight_obs=False)

model_sc.train()

DestVI.setup_anndata(adata_st, layer='counts')
model_st = DestVI.from_rna_model(adata_st, model_sc)

model_st.train(max_epochs=2500)

adata_st.obsm['proportion'] = model_st.get_proportions()
adata_st.write_h5ad("Dataset/human_lymph_node_rep1/adata_destvi.h5ad")