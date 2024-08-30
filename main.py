import os.path
import random
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
import squidpy as sq
import pandas as pd
import scanpy as sc

from spatial_integration.preprocess import clr_normalize_each_cell, fix_seed, construct_neighbor_graph, pse_srt_from_scrna, \
    ST_preprocess
from spatial_integration.SpatialGlue_pyG import Train_SpatialGlue
from spatial_integration.utils import clustering


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
spot_num = 50000
adata_pse_srt_path = 'Dataset/human_lymph_node/adata_pse_srt_' + str(spot_num) + '.h5ad'
if not os.path.exists(adata_pse_srt_path):
    adata_pse_srt = pse_srt_from_scrna(adata_scrna, spot_num=spot_num)
    adata_pse_srt.write_h5ad(adata_pse_srt_path)
else:
    adata_pse_srt = sc.read_h5ad(adata_pse_srt_path)
ground_truth = pd.read_csv('Dataset/' + dataset + '/annotation.csv') if dataset == 'human_lymph_node' else None
# Specify data type
data_type = '10x'
n_top_genes = 3000

n_comps = adata_omics2.n_vars - 1

# RNA
common_genes = [g for g in adata_omics1.var_names if g in adata_pse_srt.var_names]
adata_omics1 = adata_omics1[:, common_genes]

# preprocss ST data
adata_omics1, pca_model = ST_preprocess(adata_omics1, n_top_genes=n_top_genes, n_comps=n_comps, use_pca=False)

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
sc_included = False
data = construct_neighbor_graph(adata_omics1, adata_omics2, adata_pse_srt if sc_included else None, datatype=data_type)

tool = 'mclust'  # mclust, leiden, and louvain

model = Train_SpatialGlue(data, datatype=data_type, random_seed=random.randint(0, 10000))
# train model
output = model.train()
# %%
adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
adata.obsm['spatial_integration'] = output['spatial_integration'].copy()
adata.obsm['alpha'] = output['alpha']
adata.obsm['alpha_omics1'] = output['alpha_omics1']
adata.obsm['alpha_omics2'] = output['alpha_omics2']

clustering(adata, key='spatial_integration', add_key='spatial_integration', n_clusters=10, method=tool, use_pca=True)

prediction = adata.obs['spatial_integration']

ari = adjusted_rand_score(adata.obs['spatial_integration'], adata.obs['ground_truth'])
mi = mutual_info_score(adata.obs['spatial_integration'], adata.obs['ground_truth'])
nmi = normalized_mutual_info_score(adata.obs['spatial_integration'], adata.obs['ground_truth'])
ami = adjusted_mutual_info_score(adata.obs['spatial_integration'], adata.obs['ground_truth'])
hom = homogeneity_score(adata.obs['spatial_integration'], adata.obs['ground_truth'])
vme = v_measure_score(adata.obs['spatial_integration'], adata.obs['ground_truth'])
print('ARI score is: ' + str(ari))
print('mutual_info score is: ' + str(mi))
print('NMI score is: ' + str(nmi))
print('AMI score is: ' + str(ami))
print('homogeneity score is: ' + str(hom))
print('v_measure score is: ' + str(vme))
print('Average score is: ' + str((ari + mi + nmi + ami + hom + vme) / 6))
sq.gr.spatial_autocorr(adata, attr='obs', mode='moran', genes='spatial_integration')
print('Moran\'s I score is: ' + str(adata.uns["moranI"]['I'][0]))

a = 1
