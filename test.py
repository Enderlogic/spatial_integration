import os

import numpy as np
import pandas as pd
import scanpy as sc
from scvi_local.src.scvi.model import CondSCVI, DestVI

adata_omics1 = sc.read_h5ad('Dataset/human_lymph_node/adata_RNA.h5ad')
adata_omics1.var_names_make_unique()
sc.pp.filter_genes(adata_omics1, min_cells=10)
proportion_path = "Dataset/human_lymph_node/proportions.csv"
if not os.path.exists(proportion_path):
    adata_sc_omics1 = sc.read_h5ad('Dataset/human_lymph_node/adata_scrna.h5ad')
    adata_sc_omics1.obs['celltype'] = adata_sc_omics1.obs['Subset']
    sc.pp.filter_genes(adata_sc_omics1, min_cells=10)

    adata_sc_omics1.layers["counts"] = adata_sc_omics1.X.copy()
    sc.pp.highly_variable_genes(adata_sc_omics1, n_top_genes=3000, subset=True, layer="counts", flavor="seurat_v3")
    sc.pp.normalize_total(adata_sc_omics1, target_sum=10e4)
    sc.pp.log1p(adata_sc_omics1)
    adata_sc_omics1.raw = adata_sc_omics1

    adata_omics1.layers["counts"] = adata_omics1.X.copy()
    sc.pp.highly_variable_genes(adata_omics1, n_top_genes=3000, subset=True, layer="counts", flavor="seurat_v3")
    sc.pp.normalize_total(adata_omics1, target_sum=10e4)
    sc.pp.log1p(adata_omics1)
    adata_omics1.raw = adata_omics1
    intersect = np.intersect1d(adata_sc_omics1.var_names, adata_omics1.var_names)
    adata_omics1 = adata_omics1[:, intersect].copy()
    adata_sc_omics1 = adata_sc_omics1[:, intersect].copy()
    G = len(intersect)

    CondSCVI.setup_anndata(adata_sc_omics1, layer="counts", labels_key="celltype")
    sc_model = CondSCVI(adata_sc_omics1, weight_obs=False)
    sc_model.view_anndata_setup()
    sc_model.train()

    DestVI.setup_anndata(adata_omics1, layer="counts")
    st_model = DestVI.from_rna_model(adata_omics1, sc_model)
    st_model.view_anndata_setup()
    st_model.train(max_epochs=2500)
    proportions = st_model.get_proportions()
    proportions.to_csv(proportion_path, index=False)
else:
    proportions = pd.read_csv(proportion_path)
proportions.index = adata_omics1.obs_names
adata_omics1.obs = pd.concat([adata_omics1.obs, proportions], axis=1)
for id in range(adata_omics1.obs.columns.size):
    sc.pl.embedding(adata_omics1, basis='spatial', color=adata_omics1.obs.columns[id], s=100,
                    title=adata_omics1.obs.columns[id], save=str(adata_omics1.obs.columns[id]) + '.pdf')
a = 1
