import os
import random

from pandas import DataFrame, get_dummies
from scipy.sparse import issparse
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
import scanpy as sc
from sklearn.neighbors import kneighbors_graph

from SpatialGlue.preprocess import clr_normalize_each_cell, pca, construct_neighbor_graph, lsi
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
from SpatialGlue.utils import clustering


def moranI_score(adata, key):
    g = kneighbors_graph(adata.obsm['spatial'], 6, mode='connectivity', metric='euclidean')
    one_hot = get_dummies(adata.obs[key])
    moranI = sc.metrics.morans_i(g, one_hot.values.T).mean()
    return moranI


def calculate_jaccard(adata, adata_omics1, adata_omics2, key, k=50):
    sc.pp.neighbors(adata, use_rep=key, key_added=key, n_neighbors=k)
    sc.pp.neighbors(adata_omics1, use_rep='feat', key_added='feat', n_neighbors=k)
    sc.pp.neighbors(adata_omics2, use_rep='feat', key_added='feat', n_neighbors=k)
    jaccard1 = ((adata.obsp[key + '_distances'].toarray() * adata_omics1.obsp['feat_distances'].toarray() > 0).sum(
        1) / (adata.obsp[key + '_distances'].toarray() + adata_omics1.obsp['feat_distances'].toarray() > 0).sum(
        1)).mean()
    jaccard2 = ((adata.obsp[key + '_distances'].toarray() * adata_omics2.obsp['feat_distances'].toarray() > 0).sum(
        1) / (adata.obsp[key + '_distances'].toarray() + adata_omics2.obsp['feat_distances'].toarray() > 0).sum(
        1)).mean()
    return jaccard1, jaccard2


dataset = [n for n in os.listdir('Dataset') if os.path.isdir('Dataset/' + n)]
print(dataset)
result = DataFrame(columns=['method', 'dataset', 'metrics', 'result'])
method = 'SpatialGlue'
n_repeat = 10
for dataname in dataset:
    if 'spleen' in dataname:
        n_cluster_list = [5, 3]
        datatype = 'SPOTS'
    elif 'lymph_node' in dataname:
        n_cluster_list = [6, 10]
        datatype = '10x'
    elif 'breast_cancer' in dataname:
        n_cluster_list = [5]
        datatype = 'SPOTS'
    elif 'thymus' in dataname:
        n_cluster_list = [8]
        datatype = 'Stereo-CITE-seq'
    elif 'brain' in dataname:
        n_cluster_list = [18]
        datatype = 'Spatial-epigenome-transcriptome'
    else:
        raise Exception('Data not recognized')
    # load necessary datasets including spatial transcriptome, spatial proteome and scRNA-seq
    adata_omics1 = sc.read_h5ad('Dataset/' + dataname + '/adata_RNA.h5ad')
    adata_omics1.var_names_make_unique()
    # RNA
    sc.pp.filter_genes(adata_omics1, min_cells=10)
    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)
    adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
    if 'brain' not in dataname:
        # Protein
        adata_omics2 = sc.read_h5ad('Dataset/' + dataname + '/adata_ADT.h5ad')
        adata_omics2.var_names_make_unique()
        if issparse(adata_omics2.X):
            adata_omics2.X = adata_omics2.X.toarray()
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
    else:
        adata_omics2 = sc.read_h5ad('Dataset/' + dataname + '/adata_peaks_normalized.h5ad')
        adata_omics2.var_names_make_unique()
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)
        if 'X_lsi' not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=51)

        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
    data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=datatype)

    adata_best = adata_omics1.copy()
    adata_best.obsm['feat1'] = adata_omics1.obsm['feat']
    adata_best.obsm['feat2'] = adata_omics2.obsm['feat']
    for i in range(n_repeat):
        print(dataname, str(i + 1), 'th iteration.')
        # train model
        model = Train_SpatialGlue(data, datatype=datatype, random_seed=random.randint(0, 10000))
        output = model.train()

        # evaluation
        adata = adata_omics1.copy()
        adata.obsm[method] = output[method].copy()
        adata.obsm['feat1'] = adata_omics1.obsm['feat']
        adata.obsm['feat2'] = adata_omics2.obsm['feat']

        jaccard1, jaccard2 = calculate_jaccard(adata, adata_omics1, adata_omics2, method, k=50)
        result.loc[len(result.index)] = [method, dataname, 'jaccard1', jaccard1]
        result.loc[len(result.index)] = [method, dataname, 'jaccard2', jaccard2]
        result.loc[len(result.index)] = [method, dataname, 'jaccard', jaccard1 + jaccard2]
        for nc in n_cluster_list:
            clustering(adata, key=method, add_key=method + '_' + str(nc), n_clusters=nc, method='mclust', use_pca=True)
            prediction = adata.obs[method + '_' + str(nc)]
            new_name = method + '_' + str(nc)
            if 'cluster' in adata.obs:
                ari = adjusted_rand_score(adata.obs['cluster'], prediction)
                mi = mutual_info_score(adata.obs['cluster'], prediction)
                nmi = normalized_mutual_info_score(adata.obs['cluster'], prediction)
                ami = adjusted_mutual_info_score(adata.obs['cluster'], prediction)
                hom = homogeneity_score(adata.obs['cluster'], prediction)
                vme = v_measure_score(adata.obs['cluster'], prediction)
                ave_score = (ari + mi + nmi + ami + hom + vme) / 6

                result.loc[len(result.index)] = [new_name, dataname, 'ari', ari]
                result.loc[len(result.index)] = [new_name, dataname, 'mi', mi]
                result.loc[len(result.index)] = [new_name, dataname, 'nmi', nmi]
                result.loc[len(result.index)] = [new_name, dataname, 'ami', ami]
                result.loc[len(result.index)] = [new_name, dataname, 'hom', hom]
                result.loc[len(result.index)] = [new_name, dataname, 'vme', vme]
                result.loc[len(result.index)] = [new_name, dataname, 'average', ave_score]

            result.loc[len(result.index)] = [new_name, dataname, 'moran I', moranI_score(adata, new_name)]
result.to_csv('results/evaluation_' + method + '.csv', index=False)
