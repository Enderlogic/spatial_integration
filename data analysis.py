import matplotlib.pyplot as plt
import squidpy.gr
import scanpy as sc
import numpy as np
from GraphST.GraphST import GraphST
from GraphST.utils import project_cell_to_spot


def ST_preprocess(ST_exp, filter=True, normalize=True, log=True, highly_variable_genes=True, scale=True,
                  n_top_genes=3000):
    adata = ST_exp.copy()

    if filter:
        sc.pp.filter_genes(adata, min_cells=1)
    adata.var['mt'] = np.logical_or(adata.var_names.str.startswith('MT-'), adata.var_names.str.startswith('mt-'))
    adata.var['rb'] = adata.var_names.str.startswith(('RP', 'Rp', 'rp'))

    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    mask_cell = adata.obs['pct_counts_mt'] < 100
    mask_gene = np.logical_and(~adata.var['mt'], ~adata.var['rb'])

    adata = adata[mask_cell, mask_gene]

    if highly_variable_genes:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)

    if normalize:
        sc.pp.normalize_total(adata)

    if log:
        sc.pp.log1p(adata)

    if scale:
        sc.pp.scale(adata)
    return adata

def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata

ccRCC_merge = sc.read_h5ad('Dataset/ccRCC_ST_SM_merge_data/ccRCC_merge.h5ad')
ccRCC_merge_hvf = sc.read_h5ad('Dataset/ccRCC_ST_SM_merge_data/ccRCC_merge_hvf.h5ad')
ccRCC_merge_final = sc.read_h5ad('Dataset/ccRCC_ST_SM_merge_data/ccRCC_merge_final.h5ad')
ccRCC_merge_hvf_afterVAE = sc.read_h5ad('Dataset/ccRCC_ST_SM_merge_data/ccRCC_merge_hvf_afterVAE.h5ad')

st = sc.read_h5ad('Dataset/human_lymph_node/adata_RNA_ori.h5ad')
st.var_names_make_unique()
sc.pp.pca(st)
adt = sc.read_h5ad('Dataset/human_lymph_node/adata_ADT_ori.h5ad')
adt.var_names_make_unique()
adt.X = adt.X.toarray()
adt = clr_normalize_each_cell(adt)
sc.pp.scale(adt)

squidpy.gr.spatial_neighbors(st)
squidpy.gr.spatial_autocorr(st, mode='moran', genes=st.var_names)
squidpy.gr.spatial_autocorr(st, mode='moran', genes='X_pca', attr='obsm')

squidpy.gr.spatial_neighbors(adt)
squidpy.gr.spatial_autocorr(adt, mode='moran', genes=adt.var_names)

GraphST