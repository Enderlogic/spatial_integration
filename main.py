import argparse
import json
import os.path
import random
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
import squidpy as sq
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from spatial_integration.preprocess import clr_normalize_each_cell, construct_neighbor_graph, pse_srt_from_scrna, \
    ST_preprocess
from spatial_integration.spatial_integration import spatial_integration_train
from spatial_integration.utils import clustering

# set up some hyperparameters
parser = argparse.ArgumentParser(description='Process JSON input')
parser.add_argument('--input', '-i', type=str, help='Path to the JSON file')
args = parser.parse_args()
with open(args.input) as file:
    config = json.load(file)

default_hyper = {
    "data_type": "10x_yang",  # this parameter is used for loading pre-defined epoch number and weight factors
    "learning_rate": 1e-3,  # learning rate
    "epochs": 200,  # number of epochs
    "weight_factors": [200, 1, 1, 1, 1, 1, 1],
    # the weight factors, the first four weights are defined in the original SpatialGlue paper
    # the 5th to 8th weights correspond to data reconstruction loss, edge reconstruction loss and discrimination loss for pseudo SRT data
    "n_top_genes": 3000,  # number of highly variable genes
    "sc_included": False,  # whether to use scRNA-seq to guide spatial multiomics integration
    "tool": 'mclust',  # mclust, leiden, and louvain
    "dataset": 'human_lymph_node',  # this is the only dataset with human annotation
    "spot_num": 50000  # the number of spots in pseudo SRT data
}

hyperparameters = {key: config.get(key, default_hyper[key]) for key in default_hyper}
data_type = hyperparameters['data_type']
learning_rate = hyperparameters['learning_rate']
epochs = hyperparameters['epochs']
weight_factors = hyperparameters['weight_factors']
n_top_genes = hyperparameters['n_top_genes']
sc_included = hyperparameters['sc_included']
tool = hyperparameters['tool']
dataset = hyperparameters['dataset']
spot_num = hyperparameters['spot_num']

# load necessary datasets including spatial transcriptome, spatial proteome and scRNA-seq
adata_srt = sc.read_h5ad('Dataset/' + dataset + '/adata_RNA.h5ad')
adata_srt.var_names_make_unique()
adata_pro = sc.read_h5ad('Dataset/' + dataset + '/adata_ADT.h5ad')
adata_pro.var_names_make_unique()
adata_scrna = sc.read('Dataset/' + dataset + '/adata_scrna.h5ad',
                      backup_url='https://cell2location.cog.sanger.ac.uk/paper/integrated_lymphoid_organ_scrna/RegressionNBV4Torch_57covariates_73260cells_10237genes/sc.h5ad')
adata_scrna.obs['celltype'] = adata_scrna.obs['Subset']
ground_truth = pd.read_csv('Dataset/' + dataset + '/annotation.csv') if dataset == 'human_lymph_node' else None

# generate pseudo SRT data
adata_pse_srt_path = 'Dataset/human_lymph_node/adata_pse_srt_' + str(spot_num) + '.h5ad'
if not os.path.exists(adata_pse_srt_path):
    adata_pse_srt = pse_srt_from_scrna(adata_scrna, spot_num=spot_num)
    adata_pse_srt.write_h5ad(adata_pse_srt_path)
else:
    adata_pse_srt = sc.read_h5ad(adata_pse_srt_path)

# preprocss ST data
common_genes = [g for g in adata_srt.var_names if g in adata_pse_srt.var_names]
adata_srt = adata_srt[:, common_genes]
adata_srt = ST_preprocess(adata_srt, n_top_genes=n_top_genes)

# preprocess Protein data
adata_pro.X = adata_pro.X.toarray()
adata_pro = clr_normalize_each_cell(adata_pro)
sc.pp.scale(adata_pro)
adata_pro.obsm['feat'] = adata_pro.X.copy()

# preprocess pseudo spots
adata_pse_srt = ST_preprocess(adata_pse_srt[:, adata_srt.var_names], filter=False, highly_variable_genes=False)

# construct data
data = construct_neighbor_graph(adata_srt, adata_pro, adata_pse_srt if sc_included else None, datatype=data_type,
                                node_num=adata_srt.n_obs)

model = spatial_integration_train(data, datatype=data_type, random_seed=random.randint(0, 10000),
                                  learning_rate=learning_rate, epochs=epochs, weight_factors=weight_factors)
# train model
output = model.train()

# %% evaluation
adata = adata_srt.copy()
adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
adata.obsm['spatial_integration'] = output['spatial_integration'].copy()
adata.obsm['alpha'] = output['alpha']
adata.obsm['alpha_omics1'] = output['alpha_omics1']
adata.obsm['alpha_omics2'] = output['alpha_omics2']

clustering(adata, key='spatial_integration', add_key='spatial_integration', n_clusters=10, method=tool, use_pca=True)

prediction = adata.obs['spatial_integration']

if ground_truth is not None:
    adata.obs['ground_truth'] = ground_truth['manual-anno'].values
    ari = adjusted_rand_score(prediction, adata.obs['ground_truth'])
    mi = mutual_info_score(prediction, adata.obs['ground_truth'])
    nmi = normalized_mutual_info_score(prediction, adata.obs['ground_truth'])
    ami = adjusted_mutual_info_score(prediction, adata.obs['ground_truth'])
    hom = homogeneity_score(prediction, adata.obs['ground_truth'])
    vme = v_measure_score(prediction, adata.obs['ground_truth'])
    ave_score = (ari + mi + nmi + ami + hom + vme) / 6
    print('Average score is: ' + str(ave_score))

# TODO: verify the computation of the moran's I score
sq.gr.spatial_neighbors(adata)
sq.gr.spatial_autocorr(adata, attr='obs', mode='moran', genes='spatial_integration')
print('Moran\'s I score is: ' + str(adata.uns["moranI"]['I'][0]))

# %% visualization
figure_title = 'sc_included: ' + str(sc_included) + '\nMoran\'s I score: ' + str(round(adata.uns["moranI"]['I'][0], 5))
if ground_truth is not None:
    figure_title += '\nAverage score: ' + str(round(ave_score, 5))

fig, ax_list = plt.subplots(1, 3, figsize=(12, 3))
sc.pp.neighbors(adata, use_rep='spatial_integration', n_neighbors=10)
sc.tl.umap(adata)
sc.pl.umap(adata, color='spatial_integration', ax=ax_list[0], title=figure_title, s=20, show=False)
sc.pl.embedding(adata, basis='spatial', color='spatial_integration', ax=ax_list[1], title=figure_title, s=25,
                show=False)
sc.pl.embedding(adata, basis='spatial', color='ground_truth', ax=ax_list[2], title='Ground truth', s=25, show=False)

plt.tight_layout(w_pad=0.3)
plt.savefig('results/sc_included_' + str(sc_included) + '.pdf')
plt.show()
