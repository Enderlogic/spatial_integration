import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def generate_a_spot_passion(adata_scrna, lam, max_cell_types_in_spot, library):
    cell_num = np.random.poisson(lam=lam) + 1
    cell_type_num = random.randint(1, max_cell_types_in_spot)
    cell_type_list_selected = np.random.choice(adata_scrna.obs['celltype'].value_counts().keys(), size=cell_type_num,
                                               replace=False)
    picked_cell_type = np.unique(np.random.choice(cell_type_list_selected, size=cell_num), return_counts=True)
    picked_cells = [np.random.choice(library[picked_cell_type[0][i]], picked_cell_type[1][i], replace=False) for i in
                    range(picked_cell_type[0].size)]
    picked_cells = [x for xs in picked_cells for x in xs]
    return adata_scrna[picked_cells]


def pse_srt_from_scrna(adata_scrna, spot_num=10000, lam=5, max_cell_types_in_spot=6):
    cell_types = adata_scrna.obs['celltype'].unique()
    word_to_idx_celltype = {word: i for i, word in enumerate(cell_types)}

    cell_type_num = len(cell_types)

    generated_spots = []
    library = {i: adata_scrna[adata_scrna.obs['celltype'] == i].obs_names for i in adata_scrna.obs['celltype'].unique()}
    for _ in tqdm(range(spot_num), desc='Generating pseudo-spots'):
        generated_spots.append(generate_a_spot_passion(adata_scrna, lam, max_cell_types_in_spot, library))

    pse_srt_table = np.zeros((spot_num, adata_scrna.shape[1]), dtype=float)
    pse_fraction_table = np.zeros((spot_num, cell_type_num), dtype=float)
    for i in range(spot_num):
        one_spot = generated_spots[i]
        pse_srt_table[i] = one_spot.X.sum(axis=0)
        for j in one_spot.obs.index:
            cell_type = one_spot.obs.loc[j, 'celltype']
            type_idx = word_to_idx_celltype[cell_type]
            pse_fraction_table[i, type_idx] += 1
    pse_srt_table = pd.DataFrame(pse_srt_table, columns=adata_scrna.var.index.values)
    adata_pse_srt = sc.AnnData(X=pse_srt_table.values)
    adata_pse_srt.obs.index = pse_srt_table.index
    adata_pse_srt.var.index = pse_srt_table.columns
    pse_fraction_table = pd.DataFrame(pse_fraction_table, columns=cell_types)
    pse_fraction_table['cell_num'] = pse_fraction_table.sum(axis=1)
    for i in pse_fraction_table.columns[:-1]:
        pse_fraction_table[i] = pse_fraction_table[i] / pse_fraction_table['cell_num']
    adata_pse_srt.obs = adata_pse_srt.obs.join(pse_fraction_table)
    return adata_pse_srt


def ST_preprocess(ST_exp, filter=True, normalize=True, log=True, highly_variable_genes=True, scale=True,
                  n_top_genes=None, use_pca=False, n_comps=30, pca_model=None):
    adata = ST_exp.copy()

    if filter:
        sc.pp.filter_genes(adata, min_cells=10)

    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)

    if log:
        sc.pp.log1p(adata)

    adata.layers['scale.data'] = adata.X.copy()

    if highly_variable_genes:
        # adata = adata[:, adata.var.highly_variable]
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    if scale:
        sc.pp.scale(adata)
    if use_pca:
        if pca_model is None:
            pca_model = PCA(n_components=n_comps)
            adata.obsm['feat'] = pca_model.fit_transform(adata.X)
        else:
            adata.obsm['feat'] = pca_model.transform(adata.X)
        # sc.pp.pca(adata, n_comps=n_comps)
        return adata, pca_model
    else:
        adata.obsm['feat'] = adata.X.copy()
        return adata


class PseudoDataset(Dataset):
    def __init__(self, data, node_num, k=20):
        super(PseudoDataset, self).__init__()
        self.data = data
        self.node_num = node_num
        self.k = k
        self.num_cell_type = data.obs.shape[1] - 1
        self.feat, self.y, self.ex_adjs = self.build_graph()

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, idx):
        return self.feat[idx], self.y[idx], self.ex_adjs[idx]

    def build_graph(self):
        node_feat_ls = []
        node_y_ls = []
        adj_ls = []
        num_graphs = int(len(self.data.X) / self.node_num)
        for i in tqdm(range(num_graphs), desc='Generating pseudo-graphs'):
            node = self.data[i * self.node_num: (i + 1) * self.node_num]
            sc.pp.scale(node)
            node_feat, node_y = node.X.copy(), np.array(node.obs)[:, :-1]
            node_feat = np.nan_to_num(node_feat, nan=0, posinf=0)
            node_feat = torch.FloatTensor(node_feat.copy())
            node_y = torch.FloatTensor(node_y.copy())
            ex_adj = kneighbors_graph(node_feat, self.k, mode="connectivity", metric="correlation", include_self=False)
            ex_adj = torch.FloatTensor(ex_adj.copy().toarray())

            ex_adj = ex_adj + ex_adj.T
            ex_adj = np.where(ex_adj > 1, 1, ex_adj)

            # convert dense matrix to sparse matrix
            # ex_adj = preprocess_graph(ex_adj)  # sparse adjacent matrix corresponding to feature graph
            node_feat_ls.append(node_feat)
            node_y_ls.append(node_y)
            adj_ls.append(ex_adj)
        return node_feat_ls, node_y_ls, adj_ls


def construct_neighbor_graph(adata_omics1, adata_omics2, adata_pse_srt=None, datatype='SPOTS', n_neighbors=3,
                             node_num=200):
    """
    Construct neighbor graphs, including feature graph and spatial graph. 
    Feature graph is based expression data while spatial graph is based on cell/spot spatial coordinates.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    data : dict
        AnnData objects with preprossed data for different omics.

    """

    # construct spatial neighbor graphs
    ################# spatial graph #################
    if datatype in ['Stereo-CITE-seq', 'Spatial-epigenome-transcriptome']:
        n_neighbors = 6
        # omics1
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_omics1

    # omics2
    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = construct_graph_by_coordinate(cell_position_omics2, n_neighbors=n_neighbors)
    adata_omics2.uns['adj_spatial'] = adj_omics2

    ################# feature graph #################
    feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2)
    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = feature_graph_omics1, feature_graph_omics2

    if adata_pse_srt is not None:
        adata_pse_srt = PseudoDataset(adata_pse_srt, node_num)
    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2, 'adata_pse_srt': adata_pse_srt}

    return data


def pca(adata, use_reps=None, n_comps=10):
    """Dimension reduction with PCA algorithm"""

    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    return feat_pca


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


def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode="connectivity", metric="correlation",
                               include_self=False):
    """Constructing feature neighbor graph according to expresss profiles"""

    feature_graph_omics1 = kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric,
                                            include_self=include_self)
    feature_graph_omics2 = kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric,
                                            include_self=include_self)
    return feature_graph_omics1, feature_graph_omics2


def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    # print('n_neighbor:', n_neighbors)
    """Constructing spatial neighbor graph according to spatial coordinates."""

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj


def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def adjacent_matrix_preprocessing(adata_omics1, adata_omics2):
    """Converting dense adjacent matrix to sparse adjacent matrix"""

    ######################################## construct spatial graph ########################################
    adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
    adj_spatial_omics2 = transform_adjacent_matrix(adj_spatial_omics2)

    adj_spatial_omics1 = adj_spatial_omics1.toarray()  # To ensure that adjacent matrix is symmetric
    adj_spatial_omics2 = adj_spatial_omics2.toarray()

    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1 > 1, 1, adj_spatial_omics1)
    adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
    adj_spatial_omics2 = np.where(adj_spatial_omics2 > 1, 1, adj_spatial_omics2)

    # convert dense matrix to sparse matrix
    adj_spatial_omics1 = preprocess_graph(adj_spatial_omics1)  # sparse adjacent matrix corresponding to spatial graph
    adj_spatial_omics2 = preprocess_graph(adj_spatial_omics2)

    ######################################## construct feature graph ########################################
    adj_feature_omics1 = torch.FloatTensor(adata_omics1.obsm['adj_feature'].copy().toarray())
    adj_feature_omics2 = torch.FloatTensor(adata_omics2.obsm['adj_feature'].copy().toarray())

    adj_feature_omics1 = adj_feature_omics1 + adj_feature_omics1.T
    adj_feature_omics1 = np.where(adj_feature_omics1 > 1, 1, adj_feature_omics1)
    adj_feature_omics2 = adj_feature_omics2 + adj_feature_omics2.T
    adj_feature_omics2 = np.where(adj_feature_omics2 > 1, 1, adj_feature_omics2)

    # convert dense matrix to sparse matrix
    adj_feature_omics1 = preprocess_graph(adj_feature_omics1)  # sparse adjacent matrix corresponding to feature graph
    adj_feature_omics2 = preprocess_graph(adj_feature_omics2)
    adj = {'adj_spatial_omics1': adj_spatial_omics1, 'adj_spatial_omics2': adj_spatial_omics2,
           'adj_feature_omics1': adj_feature_omics1, 'adj_feature_omics2': adj_feature_omics2}

    return adj


def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    # X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    # adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def fix_seed(seed):
    # seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
