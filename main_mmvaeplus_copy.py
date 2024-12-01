import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import issparse
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
import scanpy as sc
import os
from SpatialGlue.preprocess import clr_normalize_each_cell
from SpatialGlue.utils import clustering
from mmvaeplus_copy.preprocess import ST_preprocess
from mmvaeplus_copy.mmvaeplus import MMVAEPLUS

dataset = [n for n in os.listdir('Dataset') if os.path.isdir('Dataset/' + n)]
# dataset = ['mouse_brain_H3K27ac']
method = 'mmvaeplus'

zs_dim = 6
zp_dim = 3
title_name = []
for dataname in dataset:
    print(dataname)
    adata = []
    recon_type = []
    for filename in os.listdir('Dataset/' + dataname):
        if filename.endswith(".h5ad"):
            adata_element = sc.read_h5ad('Dataset/' + dataname + '/' + filename)
            adata_element.var_names_make_unique()
            sc.pp.filter_genes(adata_element, min_cells=1)
            if issparse(adata_element.X):
                adata_element.X = adata_element.X.toarray()
            if adata_element.n_vars > 3000:
                adata_element = ST_preprocess(adata_element, n_top_genes=3000, n_comps=50)
                adata_element = adata_element[:, adata_element.var.highly_variable]
                recon_type.append('zinb')
            else:
                adata_element = clr_normalize_each_cell(adata_element)
                sc.pp.pca(adata_element, n_comps=adata_element.n_vars - 1)
                recon_type.append('nb')
            adata.append(adata_element)
            if 'peaks' in filename:
                title_name += ['ATAC private embedding ' + str(i + 1) for i in range(zp_dim)]
            elif 'ADT' in filename:
                title_name += ['ADT private embedding ' + str(i + 1) for i in range(zp_dim)]
            elif 'RNA' in filename:
                title_name += ['RNA private embedding ' + str(i + 1) for i in range(zp_dim)]
            if len(title_name) == zp_dim:
                title_name += ['shared embedding ' + str(i + 1) for i in range(zs_dim)]
    adata_omics1 = adata[0]
    adata_omics2 = adata[1]
    recon_type_omics1 = recon_type[0]
    recon_type_omics2 = recon_type[1]

    if 'spleen' in dataname:
        n_cluster = 5
    elif 'lymph_node' in dataname:
        n_cluster = 10
    elif 'breast_cancer' in dataname:
        n_cluster = 5
    elif 'thymus' in dataname:
        n_cluster = 6
    elif 'brain' in dataname:
        n_cluster = 18
    else:
        raise Exception('Data not recognized')
    model = MMVAEPLUS(adata_omics1, adata_omics2, n_neighbors=20, learning_rate=1e-3, epochs=200, zs_dim=zs_dim,
                      zp_dim=zp_dim, hidden_dim1=256, hidden_dim2=256, recon_type_omics1=recon_type_omics1,
                      recon_type_omics2=recon_type_omics2, weight_omics1=1, weight_omics2=1, weight_kl=10)
    # train model
    model.train(test_mode=True)
    embedding = model.encode()
    # %% evaluation
    adata = adata_omics1.copy()
    if 'cluster' in adata_omics2.obs:
        adata.obs['cluster'] = adata_omics2.obs['cluster']
    adata.obsm[method] = embedding.copy()

    adata.obsm['feat1'] = adata_omics1.obsm['X_pca']
    adata.obsm['feat2'] = adata_omics2.obsm['X_pca']
    sc.pp.neighbors(adata, use_rep=method, key_added=method, n_neighbors=51)
    sc.pp.neighbors(adata, use_rep='feat1', key_added='feat1', n_neighbors=51)
    sc.pp.neighbors(adata, use_rep='feat2', key_added='feat2', n_neighbors=51)

    clustering(adata, key=method, add_key=method, n_clusters=n_cluster, method='mclust',
               use_pca=True if adata.obsm[method].shape[1] > 20 else False)
    prediction = adata.obs[method]
    if 'cluster' in adata.obs:
        ari = adjusted_rand_score(adata.obs['cluster'], prediction)
        mi = mutual_info_score(adata.obs['cluster'], prediction)
        nmi = normalized_mutual_info_score(adata.obs['cluster'], prediction)
        ami = adjusted_mutual_info_score(adata.obs['cluster'], prediction)
        hom = homogeneity_score(adata.obs['cluster'], prediction)
        vme = v_measure_score(adata.obs['cluster'], prediction)
        ave_score = (ari + mi + nmi + ami + hom + vme) / 6

        print(dataname + ', n_cluster=' + str(n_cluster) + ', ari=' + str(ari))
    else:
        ari = np.nan

    # visualization
    s = 100
    width = 4
    height = 3
    if 'spleen' in dataname:
        adata.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata.obsm['spatial'])).T).T).T
        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
        s = 40
    elif 'breast_cancer' in dataname:
        adata.obsm['spatial'][:, 0] = -1 * adata.obsm['spatial'][:, 0]
        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
        height = 5
        s = 100
    elif 'thymus' in dataname:
        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
        s = 50
    elif 'brain' in dataname:
        s = 40
    if 'cluster' in adata.obs:
        fig, ax_list = plt.subplots(1, 2, figsize=(width * 2, height))
        sc.pl.embedding(adata, basis='spatial', color='cluster', ax=ax_list[0], title='Ground truth', s=s, show=False)

        figure_title = method + ' (' + str(n_cluster) + ' clusters)\nARI: ' + str(round(ari, 5))
        sc.pl.embedding(adata, basis='spatial', color=method, ax=ax_list[1], title=figure_title, s=s, show=False)
    else:
        sc.pl.embedding(adata, basis='spatial', color=method, s=s, show=False)

    plt.tight_layout(w_pad=0.3)
    plt.show()

    adata.obs[list(range(adata.obsm[method].shape[1]))] = adata.obsm[method]
    fig, axes = plt.subplots(3, 4, figsize = (width * 4, height * 3))
    for i in range(adata.obsm[method].shape[1]):
        sc.pl.embedding(adata, basis='spatial', color = [i], s = s, show=False, title=title_name[i], ax=axes[i % 3, i // 3])
    plt.tight_layout(w_pad=0.3)
    plt.savefig('figures/' + dataname + '.pdf')
    a = 1
