import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import issparse
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
import scanpy as sc

from SpatialGlue.preprocess import clr_normalize_each_cell
from SpatialGlue.utils import clustering
from mmvaeplus_scrna.preprocess import ST_preprocess, pse_srt_from_scrna
from mmvaeplus_scrna.mmvaeplus import MMVAEPLUS

# dataset = ['mouse_spleen_rep2', 'human_lymph_node', 'mouse_spleen_rep1', 'mouse_breast_cancer']
dataset = ['human_lymph_node']
method = 'mmvaeplus'
n_cluster = 10
lam = 2
max_cell_types_in_spot = 4
n_neighbors = 20
learning_rate = 1e-3
zs_dim = 32
zp_dim = 32
hidden_dim1 = 256
hidden_dim2 = 256
weight_omics1 = 1
weight_kl = 10
recon_type_omics1 = 'zinb'
recon_type_omics2 = 'nb'
heads = 1
n_batches = 9
weight_pse_omics1 = .1
weight_dis = 500
weight_clas = 0
# weight_pse_omics1 = 0
# weight_dis = 0
# weight_clas = 0
result_path = 'results/test_wrtscrna.csv'
result = pd.DataFrame(
    columns=['dataset', 'learning_rate', 'zs_dim', 'zp_dim', 'hidden_dim1', 'hidden_dim2', 'weight_omics1',
             'weight_omics2', 'weight_kl', 'epoch', 'n_cluster', 'ari', 'mi', 'nmi', 'ami', 'hom', 'vme', 'ave_score'])
result.to_csv(result_path, index=False)
for dataname in dataset:
    if dataname == 'human_lymph_node':
        epochs = 150
    elif dataname == 'mouse_breast_cancer':
        epochs = 50
    else:
        epochs = 600
    # load necessary datasets including spatial transcriptome and spatial proteome
    adata_omics1 = sc.read_h5ad('Dataset/' + dataname + '/adata_RNA.h5ad')
    adata_omics1.var_names_make_unique()
    adata_omics2 = sc.read_h5ad('Dataset/' + dataname + '/adata_ADT.h5ad')
    adata_omics2.var_names_make_unique()
    sc.pp.filter_genes(adata_omics1, min_cells=1)
    adata_sc_omics1 = sc.read_h5ad('Dataset/' + dataname + '/adata_scrna.h5ad')
    adata_sc_omics1.obs['celltype'] = adata_sc_omics1.obs['Subset']

    weight_omics2 = adata_omics1.n_vars / adata_omics2.n_vars
    if 'mouse_spleen' in dataname:
        n_cluster_list = [5, 3]
    elif dataname == 'human_lymph_node':
        n_cluster_list = [6, 10]
    elif dataname == 'mouse_breast_cancer':
        n_cluster_list = [5]
    else:
        raise Exception('Data not recognized')
    # adata_sc_omics1=None
    model = MMVAEPLUS(adata_omics1, adata_omics2, adata_sc_omics1, n_batches=n_batches, n_neighbors=n_neighbors,
                      learning_rate=learning_rate, epochs=600, zs_dim=zs_dim, zp_dim=zp_dim, hidden_dim1=hidden_dim1,
                      hidden_dim2=hidden_dim2, weight_omics1=weight_omics1, weight_omics2=weight_omics2,
                      weight_kl=weight_kl, weight_pse_omics1=weight_pse_omics1, weight_dis=weight_dis,
                      weight_clas=weight_clas, recon_type_omics1=recon_type_omics1, recon_type_omics2=recon_type_omics2,
                      heads=heads)
    # train model
    model.train(dataset=dataname, test_mode=True, n_cluster_list=n_cluster_list, result_path=result_path)
    embedding = model.encode()

    outputs = model.generation()
    # %% evaluation
    adata = adata_omics1.copy()
    adata.obsm[method] = embedding.copy()

    adata.obsm['feat1'] = adata_omics1.obsm['X_pca']
    adata.obsm['feat2'] = adata_omics2.obsm['X_pca']
    sc.pp.neighbors(adata, use_rep=method, key_added=method, n_neighbors=51)
    sc.pp.neighbors(adata, use_rep='feat1', key_added='feat1', n_neighbors=51)
    sc.pp.neighbors(adata, use_rep='feat2', key_added='feat2', n_neighbors=51)

    clustering(adata, key=method, add_key=method, n_clusters=n_cluster, method='mclust', use_pca=True)
    prediction = adata.obs[method]
    ari = adjusted_rand_score(adata.obs['cluster'], prediction)
    mi = mutual_info_score(adata.obs['cluster'], prediction)
    nmi = normalized_mutual_info_score(adata.obs['cluster'], prediction)
    ami = adjusted_mutual_info_score(adata.obs['cluster'], prediction)
    hom = homogeneity_score(adata.obs['cluster'], prediction)
    vme = v_measure_score(adata.obs['cluster'], prediction)
    ave_score = (ari + mi + nmi + ami + hom + vme) / 6

    print(dataname + ', n_cluster=' + str(n_cluster) + ', ari=' + str(ari))

    # visualization
    s = 25
    width = 4
    height = 3
    if 'mouse_spleen' in dataname:
        adata.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata.obsm['spatial'])).T).T).T
        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
        s = 30
    elif dataname == 'mouse_breast_cancer':
        adata.obsm['spatial'][:, 0] = -1 * adata.obsm['spatial'][:, 0]
        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
        height = 5
        s = 90

    fig, ax_list = plt.subplots(1, 2, figsize=(width * 2, height))
    sc.pl.embedding(adata, basis='spatial', color='cluster', ax=ax_list[0], title='Ground truth', s=s, show=False)

    figure_title = method + ' (' + str(n_cluster) + ' clusters)\nARI: ' + str(round(ari, 5))
    sc.pl.embedding(adata, basis='spatial', color=method, ax=ax_list[1], title=figure_title, s=s, show=False)

    plt.tight_layout(w_pad=0.3)
    plt.show()
    a = 1
