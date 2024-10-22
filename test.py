import numpy as np
import pandas
import pandas as pd
import scanpy as sc
from anndata import read_h5ad
from matplotlib import pyplot as plt
from matplotlib.pyplot import title

from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score

adata = read_h5ad('Dataset/human_lymph_node/adata_hln.h5ad')

adata.obs = adata.obs[adata.obs.columns[4:-1]]
adata.obs = adata.obs.div(adata.obs.sum(axis=1), axis=0)
for ct in adata.obs.columns:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sc.pl.embedding(adata, basis='spatial', color=ct, ax=ax, s=50, title=ct, show=False)
    plt.savefig('figures/show' + ct + '.pdf', bbox_inches='tight')


# adata_best = read_h5ad('results/mmvaeplus_human_lymph_node_best.h5ad')
# result = pandas.read_csv('results/evaluation_mmvaeplus.csv')
# ground_truth = pd.read_csv('Dataset/human_lymph_node/annotation.csv')['manual-anno'].values
# fig, ax_list = plt.subplots(1, 3, figsize=(12, 3))
# sc.pl.embedding(adata_best, basis='spatial', color='ground_truth', ax=ax_list[0], title='Ground truth', s=25,
#                 show=False)
# figure_title = 'mmvaeplus (6 clusters)\nARI score: ' + str(
#     round(result[(result['method'] == 'mmvaeplus_6') & (result['metrics'] == 'ari')]['result'].max(), 5))
# sc.pl.embedding(adata_best, basis='spatial', color='mmvaeplus_6', ax=ax_list[1], title=figure_title, s=25,
#                 show=False)
# figure_title = 'mmvaeplus (10 clusters)\nARI score: ' + str(
#     round(result[(result['method'] == 'mmvaeplus_10') & (result['metrics'] == 'ari')]['result'].max(), 5))
# sc.pl.embedding(adata_best, basis='spatial', color= 'mmvaeplus_10', ax=ax_list[2], title=figure_title, s=25,
#                 show=False)
# plt.tight_layout(w_pad=0.3)
# plt.savefig('results/mmvaeplus_human_lymph_node_best.pdf')
a = 1