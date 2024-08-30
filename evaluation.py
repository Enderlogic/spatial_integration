import os.path
import random

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
import squidpy as sq
import pandas as pd
import scanpy as sc
from tqdm import tqdm

from spatial_integration.preprocess import clr_normalize_each_cell, fix_seed, construct_neighbor_graph
from spatial_integration.SpatialGlue_pyG import Train_SpatialGlue
from spatial_integration.utils import clustering


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
                  n_top_genes=None, use_pca=True, n_comps=30, pca_model=None):
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
        return adata


dataset = 'human_lymph_node'
adata_omics1 = sc.read_h5ad('Dataset/' + dataset + '/adata_RNA.h5ad')
adata_omics1.var_names_make_unique()
if dataset in ['human_lymph_node', 'mouse_thymus_Stereo-CITE-seq']:
    adata_omics2 = sc.read_h5ad('Dataset/' + dataset + '/adata_ADT.h5ad')
elif dataset in ['Mouse_Brain_ATAC']:
    adata_omics2 = sc.read_h5ad('Dataset/' + dataset + '/adata_peaks_normalized.h5ad')
else:
    raise ValueError('Dataset not found')
adata_omics2.var_names_make_unique()
adata_scrna = sc.read_h5ad('Dataset/' + dataset + '/adata_scrna.h5ad')
adata_scrna.obs['celltype'] = adata_scrna.obs['Subset']
# pseudo spots
spot_num = 5000
data_type = '10x_yang'
adata_pse_srt_path = 'Dataset/human_lymph_node/adata_pse_srt_' + str(spot_num) + '.h5ad'
if not os.path.exists(adata_pse_srt_path):
    adata_pse_srt = pse_srt_from_scrna(adata_scrna, spot_num=spot_num)
    adata_pse_srt.write_h5ad(adata_pse_srt_path)
else:
    adata_pse_srt = sc.read_h5ad(adata_pse_srt_path)
ground_truth = pd.read_csv('Dataset/' + dataset + '/annotation.csv') if dataset == 'human_lymph_node' else None

# random_seed = 2022
# fix_seed(random_seed)
n_top_genes = 3000

n_comps = adata_omics2.n_vars - 1

# RNA
common_genes = [g for g in adata_omics1.var_names if g in adata_pse_srt.var_names]
adata_omics1 = adata_omics1[:, common_genes]

# preprocss ST data
adata_omics1, pca_model = ST_preprocess(adata_omics1, n_top_genes=n_top_genes, n_comps=n_comps)

# Protein
adata_omics2.X = adata_omics2.X.toarray()
adata_omics2 = clr_normalize_each_cell(adata_omics2)
sc.pp.scale(adata_omics2)
sc.pp.pca(adata_omics2, n_comps=n_comps)
adata_omics2.obsm['feat'] = adata_omics2.obsm['X_pca']

adata = adata_omics1.copy()
ground_truth = pd.read_csv('Dataset/human_lymph_node/annotation.csv')
adata.obs['ground_truth'] = ground_truth['manual-anno'].values
sq.gr.spatial_neighbors(adata)

# preprocess pseudo spots
adata_pse_srt, _ = ST_preprocess(adata_pse_srt[:, adata_omics1.var_names], filter=False, n_top_genes=n_top_genes,
                                 n_comps=n_comps, pca_model=pca_model)
tool = 'mclust'  # mclust, leiden, and louvain

for sc_included in [True, False]:
    # Specify data type
    # data_type = '10x_yang' if sc_included else '10x'
    data = construct_neighbor_graph(adata_omics1, adata_omics2, adata_pse_srt if sc_included else None,
                                    datatype=data_type)
    result_score = []
    result_moran = []
    for _ in range(10):
        model = Train_SpatialGlue(data, datatype=data_type, random_seed=random.randint(0, 10000), learning_rate=1e-3,
                                  epochs=200, weight_factors=[200, 1, 1, 1])
        # train model
        output = model.train()
        adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
        adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
        adata.obsm['spatial_integration'] = output['spatial_integration'].copy()
        adata.obsm['alpha'] = output['alpha']
        adata.obsm['alpha_omics1'] = output['alpha_omics1']
        adata.obsm['alpha_omics2'] = output['alpha_omics2']

        clustering(adata, key='spatial_integration', add_key='spatial_integration', n_clusters=6, method=tool, use_pca=True)

        prediction = adata.obs['spatial_integration']

        ari = adjusted_rand_score(adata.obs['spatial_integration'], adata.obs['ground_truth'])
        mi = mutual_info_score(adata.obs['spatial_integration'], adata.obs['ground_truth'])
        nmi = normalized_mutual_info_score(adata.obs['spatial_integration'], adata.obs['ground_truth'])
        ami = adjusted_mutual_info_score(adata.obs['spatial_integration'], adata.obs['ground_truth'])
        hom = homogeneity_score(adata.obs['spatial_integration'], adata.obs['ground_truth'])
        vme = v_measure_score(adata.obs['spatial_integration'], adata.obs['ground_truth'])
        print('Average score is: ' + str((ari + mi + nmi + ami + hom + vme) / 6))
        sq.gr.spatial_autocorr(adata, attr='obs', mode='moran', genes='spatial_integration')
        result_score.append((ari + mi + nmi + ami + hom + vme) / 6)
        result_moran.append(adata.uns["moranI"]['I'][0])
    print("Method is: " + "yang" if sc_included else "original")
    print('score mean is: ' + str(np.array(result_score).mean()) + ' std is: ' + str(np.array(result_score).std()))
    print('moran mean is: ' + str(np.array(result_moran).mean()) + ' std is: ' + str(np.array(result_moran).std()))
    if sc_included:
        result_mean_yang = np.array(result_score).mean()
        result_std_yang = np.array(result_score).std()
    else:
        result_mean_ori = np.array(result_score).mean()
        result_std_ori = np.array(result_score).std()
a = 1
