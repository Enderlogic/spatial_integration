import pickle

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.sparse import issparse
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
import scanpy as sc

from SpatialGlue.preprocess import clr_normalize_each_cell
from SpatialGlue.utils import clustering
from mmvaeplus.preprocess import ST_preprocess
from mmvaeplus.mmvaeplus import MMVAEPLUS

sc.set_figure_params(fontsize=22)
# dataset = ['mouse_breast_cancer', 'human_lymph_node_rep1', 'mouse_spleen_rep1', 'mouse_spleen_rep2']
dataset = ['human_lymph_node_rep1']
result = DataFrame(columns=['dataset', 'embedding_type', 'metrics', 'result'])
n_repeat = 10
embedding_types = ['z', 'zs', 'z_omics1', 'zs_omics1', 'zp_omics1', 'z_omics2', 'zs_omics2', 'zp_omics2']
for dataname in dataset:
    if dataname == 'human_lymph_node_rep1':
        n_cluster = 10
        epochs = 150
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
    best_ari = 0
    for i in range(n_repeat):
        model = MMVAEPLUS(adata_omics1, adata_omics2, n_neighbors=20, learning_rate=1e-3, epochs=epochs, zs_dim=32,
                          zp_dim=32, hidden_dim1=256, hidden_dim2=256, weight_omics1=1, weight_omics2=1,
                          recon_type='zinb', heads=1, weight_kl=10)
        # train model
        model.train(test_mode=False)
        embedding = model.encode_test()
        # evaluation
        adata = adata_omics1.copy()
        for et in embedding_types:
            adata.obsm[et] = embedding[et].copy()
            clustering(adata, key=et, add_key=et, n_clusters=n_cluster, method='mclust',
                       use_pca=True if adata.obsm[et].shape[1] > 20 else False)
            prediction = adata.obs[et]
            ari = adjusted_rand_score(adata.obs['cluster'], prediction)
            mi = mutual_info_score(adata.obs['cluster'], prediction)
            nmi = normalized_mutual_info_score(adata.obs['cluster'], prediction)
            ami = adjusted_mutual_info_score(adata.obs['cluster'], prediction)
            hom = homogeneity_score(adata.obs['cluster'], prediction)
            vme = v_measure_score(adata.obs['cluster'], prediction)
            ave_score = (ari + mi + nmi + ami + hom + vme) / 6
            if et == 'z' and ari > best_ari:
                for et2 in embedding_types:
                    adata_best.obsm[et2] = embedding[et2].copy()
                best_ari = ari
            result.loc[len(result.index)] = [dataname, et, 'ari', ari]
            result.loc[len(result.index)] = [dataname, et, 'mi', mi]
            result.loc[len(result.index)] = [dataname, et, 'nmi', nmi]
            result.loc[len(result.index)] = [dataname, et, 'ami', ami]
            result.loc[len(result.index)] = [dataname, et, 'hom', hom]
            result.loc[len(result.index)] = [dataname, et, 'vme', vme]
            result.loc[len(result.index)] = [dataname, et, 'average', ave_score]
            print(dataname + ' ' + str(i) + 'th iteration, embedding type is: ' + et + ', ari=' + str(ari))
    # visualization
    s = 100
    width = 7
    height = 5
    if 'mouse_spleen' in dataname:
        adata_best.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata_best.obsm['spatial'])).T).T).T
        adata_best.obsm['spatial'][:, 1] = -1 * adata_best.obsm['spatial'][:, 1]
        s = 110
    elif dataname == 'mouse_breast_cancer':
        adata_best.obsm['spatial'][:, 0] = -1 * adata_best.obsm['spatial'][:, 0]
        adata_best.obsm['spatial'][:, 1] = -1 * adata_best.obsm['spatial'][:, 1]
        height = 7
        width = 8
        s = 250
    elif dataname == 'human_lymph_node_rep1':
        height = 6
        width = 8

    fig, ax_list = plt.subplots(3, 3, figsize=(width * 3, height * 3))
    sc.pl.embedding(adata_best, basis='spatial', color='cluster', ax=ax_list[0, 0], title='Ground truth', s=s,
                    show=False)
    i = 1
    for et in embedding_types:
        clustering(adata_best, key=et, add_key=et, n_clusters=n_cluster, method='mclust',
                   use_pca=True if adata_best.obsm[et].shape[1] > 20 else False)
        prediction = adata_best.obs[et]
        ari = adjusted_rand_score(adata_best.obs['cluster'], prediction)
        figure_title = et + '\nBest ARI in ' + str(n_repeat) + ' repeats: ' + str(round(
            result[(result['embedding_type'] == et) & (result['metrics'] == 'ari') & (result['dataset'] == dataname)][
                'result'].max(), 4))
        row_id = i // 3
        col_id = i % 3
        sc.pl.embedding(adata_best, basis='spatial', color=et, ax=ax_list[row_id, col_id], title=figure_title, s=s,
                        show=False)
        i += 1

    plt.tight_layout(w_pad=0.3)
    plt.savefig('results/embedding_types_test' + '_' + dataname + '_best.pdf')
    adata_best.write_h5ad('results/mmvaeplus' + '_' + dataname + '_best_et.h5ad')
    # plt.show()
result.to_csv('results/evaluation_embedding_types_test.csv', index=False)
