import itertools
import os
import sys
import pandas as pd
import scanpy as sc
from pandas import DataFrame

from mmvaeplus.preprocess import clr_normalize_each_cell, pse_srt_from_scrna, ST_preprocess
from mmvaeplus.mmvaeplus import MMVAEPLUS

learning_rate_list = [1e-3]
zs_dim_list = [16]
zp_dim_list = [16]
hidden_dim1_list = [256]
hidden_dim2_list = [256]
weight_omics1_list = [1]
weight_omics2_list = [100]
weight_kl_list = [10]
weight_dis_list = [1, 10, 20]
weight_clas_list = [1, 10, 20]
pretrain_ratio_list = [1, .5, 0]

pss = list(itertools.product(
    *[learning_rate_list, zs_dim_list, zp_dim_list, hidden_dim1_list, hidden_dim2_list, weight_omics1_list,
      weight_omics2_list, weight_kl_list, weight_dis_list, weight_clas_list, pretrain_ratio_list]))

# load necessary datasets including spatial transcriptome and spatial proteome
adata_srt = sc.read_h5ad('Dataset/human_lymph_node/adata_RNA.h5ad')
adata_srt.var_names_make_unique()
sc.pp.filter_genes(adata_srt, min_cells=1)
adata_spr = sc.read_h5ad('Dataset/human_lymph_node/adata_ADT.h5ad')

# load scRNA data
adata_scrna = sc.read_h5ad('Dataset/human_lymph_node/adata_scrna.h5ad')
adata_scrna.var_names_make_unique()
common_genes = [g for g in adata_srt.var_names if g in adata_scrna.var_names]
adata_srt = adata_srt[:, common_genes]
adata_scrna = adata_scrna[:, common_genes]
adata_scrna.obs['celltype'] = adata_scrna.obs['Subset']

# preprocss ST data
adata_srt = ST_preprocess(adata_srt, n_top_genes=3000, n_comps=50)
adata_srt = adata_srt[:, adata_srt.var.highly_variable]

# preprocess Protein data
adata_spr.X = adata_spr.X.toarray()
adata_spr = clr_normalize_each_cell(adata_spr)
sc.pp.pca(adata_spr, n_comps=adata_spr.n_vars - 1)

start_idx = int(sys.argv[1])
total_idx = int(sys.argv[2])

result_path = 'results/parameter_search_vae.csv'
if not os.path.exists(result_path):
    result = DataFrame(
        columns=["learning_rate", "zs_dim", "zp_dim", "hidden_dim1", "hidden_dim2", "weight_omics1", "weight_omics2",
                 "weight_kl", "weight_dis", "weight_clas", "pretrain_ratio", "spot_num", "epoch", "ari", "mi", "nmi",
                 "ami", "hom", "vme", "ave_score"])
    result.to_csv(result_path, index=False)

count = 0
for i in range(int(start_idx * len(pss) / total_idx), int((start_idx + 1) * len(pss) / total_idx)):
    learning_rate, zs_dim, zp_dim, hidden_dim1, hidden_dim2, weight_omics1, weight_omics2, weight_kl, weight_dis, weight_clas, pretrain_ratio = \
        pss[i]
    if pretrain_ratio == 1:
        if weight_dis != weight_dis_list[0] or weight_clas != weight_clas_list[0]:
            continue
    pse_data_file = 'Dataset/human_lymph_node/adata_pse_omics1_50000.h5ad'
    if os.path.exists(pse_data_file):
        adata_pse = sc.read_h5ad(pse_data_file)
    else:
        adata_pse = pse_srt_from_scrna(adata_scrna, spot_num=adata_srt.n_obs)
        adata_pse = ST_preprocess(adata_pse, highly_variable_genes=False, n_comps=50)
        adata_pse.obs = adata_pse.obs[adata_scrna.obs['celltype'].unique()]
        adata_pse = adata_pse[:, adata_srt.var_names]
        adata_pse.write_h5ad(pse_data_file)
    for _ in range(10):
        model = MMVAEPLUS(adata_srt, adata_spr, adata_pse, n_neighbors=20, learning_rate=learning_rate,
                          epochs=200, zs_dim=zs_dim, zp_dim=zp_dim, hidden_dim1=hidden_dim1,
                          hidden_dim2=hidden_dim2, weight_omics1=weight_omics1,
                          weight_omics2=weight_omics2, weight_kl=weight_kl, weight_dis=weight_dis,
                          weight_clas=weight_clas, pretrain_ratio=pretrain_ratio)
        # train model
        model.train(ground_truth, result_path=result_path, spot_num=50000)
