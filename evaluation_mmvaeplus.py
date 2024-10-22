import os

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
import scanpy as sc

from SpatialGlue.preprocess import clr_normalize_each_cell
from SpatialGlue.utils import clustering
from mmvaeplus_scrna.preprocess import ST_preprocess, pse_srt_from_scrna
from mmvaeplus_scrna.mmvaeplus import MMVAEPLUS

# dataset = ['human_lymph_node', 'mouse_spleen_rep1', 'mouse_spleen_rep2', 'mouse_thymus']
dataset = ['human_lymph_node']
result = DataFrame(columns=['method', 'dataset', 'metrics', 'result'])
method = 'mmvaeplus'
result_path = 'results/test_wrtscrna.csv'
n_repeat = 10

for dataname in dataset:
    if dataname == 'human_lymph_node':
        epochs = 150
    elif dataname == 'mouse_breast_cancer':
        epochs = 50
    else:
        epochs = 600
    # load necessary datasets including spatial transcriptome, spatial proteome and scRNA-seq
    adata_omics1 = sc.read_h5ad('Dataset/' + dataname + '/adata_RNA.h5ad')
    adata_omics1.var_names_make_unique()
    adata_omics2 = sc.read_h5ad('Dataset/' + dataname + '/adata_ADT.h5ad')
    adata_omics2.var_names_make_unique()
    sc.pp.filter_genes(adata_omics1, min_cells=1)

    # load scRNA data
    adata_scrna = sc.read_h5ad('Dataset/' + dataname + '/adata_scrna.h5ad')
    adata_scrna.var_names_make_unique()
    common_genes = [g for g in adata_omics1.var_names if g in adata_scrna.var_names]
    adata_omics1 = adata_omics1[:, common_genes]
    adata_scrna = adata_scrna[:, common_genes]
    adata_scrna.obs['celltype'] = adata_scrna.obs['Subset']

    # preprocss ST data
    adata_omics1 = ST_preprocess(adata_omics1, n_top_genes=3000, n_comps=50)

    # pse_omics1_filename = 'Dataset/' + dataname + '/adata_pse_omics1_50000.h5ad'
    # if os.path.isfile(pse_omics1_filename):
    #     adata_pse_omics1 = sc.read(pse_omics1_filename)
    # else:
    #     adata_pse_omics1 = pse_srt_from_scrna(adata_scrna, spot_num=50000)
    #     adata_pse_omics1 = ST_preprocess(adata_pse_omics1, highly_variable_genes=False, n_comps=50)
    #     adata_pse_omics1.obs = adata_pse_omics1.obs[adata_scrna.obs['celltype'].unique()]
    #     adata_pse_omics1 = adata_pse_omics1[:, adata_omics1.var.highly_variable]
    #     adata_pse_omics1.write(pse_omics1_filename)
    adata_omics1 = adata_omics1[:, adata_omics1.var.highly_variable]

    # preprocess Protein data
    adata_omics2.X = adata_omics2.X.toarray()
    adata_omics2 = clr_normalize_each_cell(adata_omics2)
    sc.pp.pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)

    adata_best = adata_omics1.copy()
    if 'mouse_spleen' in dataname:
        n_cluster_list = [5, 3]
    elif dataname == 'human_lymph_node':
        n_cluster_list = [6, 10]
    elif dataname == 'mouse_breast_cancer':
        n_cluster_list = [5]
    else:
        raise Exception('Data not recognized')
    for i in range(n_repeat):
        model = MMVAEPLUS(adata_omics1, adata_omics2, None, n_neighbors=20, learning_rate=1e-3, epochs=150, zs_dim=32,
                          zp_dim=32, hidden_dim1=256, hidden_dim2=256, weight_omics1=1,
                          weight_omics2=adata_omics1.n_vars / adata_omics2.n_vars, weight_kl=10, weight_dis=0,
                          weight_clas=0, recon_type_omics1='zinb', recon_type_omics2='nb', heads=1)
        # train model
        model.train(test_mode=False)

        embedding = model.encode()
        # %% evaluation
        adata = adata_omics1.copy()
        adata.obsm[method] = embedding.copy()

        # adata.obsm['feat1'] = adata_omics1.obsm['X_pca']
        # adata.obsm['feat2'] = adata_omics2.X
        # sc.pp.neighbors(adata, use_rep=method, key_added=method, n_neighbors=51)
        # sc.pp.neighbors(adata, use_rep='feat1', key_added='feat1', n_neighbors=51)
        # sc.pp.neighbors(adata, use_rep='feat2', key_added='feat2', n_neighbors=51)
        #
        # result.loc[len(result.index)] = [method, dataname, 'jaccard1', (
        #         (adata.obsp[method + '_distances'].toarray() * adata.obsp['feat1_distances'].toarray() > 0).sum(
        #             1) / (adata.obsp[method + '_distances'].toarray() + adata.obsp[
        #     'feat1_distances'].toarray() > 0).sum(1)).mean()]
        # result.loc[len(result.index)] = [method, dataname, 'jaccard2', (
        #         (adata.obsp[method + '_distances'].toarray() * adata.obsp['feat2_distances'].toarray() > 0).sum(
        #             1) / (adata.obsp[method + '_distances'].toarray() + adata.obsp[
        #     'feat2_distances'].toarray() > 0).sum(1)).mean()]
        # result.loc[len(result.index)] = [method, dataname, 'jaccard',
        #                                  result[result['metrics'] == 'jaccard1']['result'].iloc[-1] +
        #                                  result[result['metrics'] == 'jaccard2']['result'].iloc[-1]]
        for nc in n_cluster_list:
            clustering(adata, key=method, add_key=method + '_' + str(nc), n_clusters=nc, method='mclust',
                       use_pca=True)
            prediction = adata.obs[method + '_' + str(nc)]
            ari = adjusted_rand_score(adata_best.obs['cluster'], prediction)
            mi = mutual_info_score(adata_best.obs['cluster'], prediction)
            nmi = normalized_mutual_info_score(adata_best.obs['cluster'], prediction)
            ami = adjusted_mutual_info_score(adata_best.obs['cluster'], prediction)
            hom = homogeneity_score(adata_best.obs['cluster'], prediction)
            vme = v_measure_score(adata_best.obs['cluster'], prediction)
            ave_score = (ari + mi + nmi + ami + hom + vme) / 6
            new_name = method + '_' + str(nc)
            if ari > result[
                (result['method'] == new_name) & (result['metrics'] == 'ari') & (result['dataset'] == dataname)][
                'result'].max() or i == 0:
                adata_best.obs[new_name] = prediction

            result.loc[len(result.index)] = [new_name, dataname, 'ari', ari]
            result.loc[len(result.index)] = [new_name, dataname, 'mi', mi]
            result.loc[len(result.index)] = [new_name, dataname, 'nmi', nmi]
            result.loc[len(result.index)] = [new_name, dataname, 'ami', ami]
            result.loc[len(result.index)] = [new_name, dataname, 'hom', hom]
            result.loc[len(result.index)] = [new_name, dataname, 'vme', vme]
            result.loc[len(result.index)] = [new_name, dataname, 'average', ave_score]
            print(dataname + ' ' + str(i) + 'th iteration, n_cluster=' + str(nc) + ', ari=' + str(ari))

    # visualization
    s = 25
    width = 4
    height = 3
    if 'mouse_spleen' in dataname:
        adata_best.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata_best.obsm['spatial'])).T).T).T
        adata_best.obsm['spatial'][:, 1] = -1 * adata_best.obsm['spatial'][:, 1]
        s = 30
    elif dataname == 'mouse_breast_cancer':
        adata_best.obsm['spatial'][:, 0] = -1 * adata_best.obsm['spatial'][:, 0]
        adata_best.obsm['spatial'][:, 1] = -1 * adata_best.obsm['spatial'][:, 1]
        height = 5
        s = 90

    fig, ax_list = plt.subplots(1, len(n_cluster_list) + 1, figsize=(width * (len(n_cluster_list) + 1), height))
    sc.pl.embedding(adata_best, basis='spatial', color='cluster', ax=ax_list[0], title='Ground truth', s=s,
                    show=False)
    i = 1
    for nc in n_cluster_list:
        figure_title = method + ' (' + str(nc) + ' clusters)\nBest ARI in ' + str(n_repeat) + ' repeats: ' + str(
            round(
                result[(result['method'] == method + '_' + str(nc)) & (result['metrics'] == 'ari') & (
                        result['dataset'] == dataname)]['result'].max(), 5))
        sc.pl.embedding(adata_best, basis='spatial', color=method + '_' + str(nc), ax=ax_list[i],
                        title=figure_title, s=s, show=False)
        i += 1
    plt.tight_layout(w_pad=0.3)
    plt.savefig('results/' + method + '_' + dataname + '_best.pdf')
    # plt.show()
    adata_best.write_h5ad('results/' + method + '_' + dataname + '_best.h5ad')
result.to_csv('results/evaluation_' + method + '_scrna.csv', index=False)
