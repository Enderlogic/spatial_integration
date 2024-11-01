from datetime import datetime

import numpy as np
import pandas
import scanpy
import torch
from anndata import AnnData
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame
from scipy.sparse import issparse
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import scanpy as sc

from .utils import clustering
from .model import mmvaeplus, NegBinom
from .preprocess import adjacent_matrix_preprocessing, ST_preprocess, clr_normalize_each_cell


class MMVAEPLUS:
    def __init__(self, adata_omics1, adata_omics2, adata_sc_omics1=None, n_significant_genes=30, n_neighbors=20,
                 learning_rate=1e-3, weight_decay=0.00, weight_omics1=1, weight_omics2=1, weight_kl=1, weight_clas=1,
                 recon_type_omics1='nb', recon_type_omics2='nb', heads=1, epochs=600, zs_dim=32, zp_dim=32,
                 hidden_dim1=256, hidden_dim2=64, verbose=True):
        # preprocss ST data
        self.epoch = 0
        adata_omics1 = ST_preprocess(adata_omics1, n_top_genes=3000, n_comps=50)

        if adata_sc_omics1 is not None:
            sig_genes = []
            common_genes = [g for g in adata_sc_omics1.var_names if g in adata_omics1.var_names]
            adata_sc_omics1 = adata_sc_omics1[:, common_genes]
            sc.pp.log1p(adata_sc_omics1)
            sc.tl.rank_genes_groups(adata_sc_omics1, 'celltype', use_raw=False)
            sig_score = pandas.DataFrame(columns=adata_sc_omics1.obs['celltype'].unique())
            for ct in adata_sc_omics1.obs['celltype'].unique():
                sig_genes += list(adata_sc_omics1.uns['rank_genes_groups']['names'][ct][: n_significant_genes])
                sc.tl.score_genes(adata_omics1,
                                  adata_sc_omics1.uns['rank_genes_groups']['names'][ct][: n_significant_genes])
                sig_score[ct] = adata_omics1.obs['score']
            sig_score[sig_score < 0] = 1e-6
            sig_score = sig_score.div(sig_score.sum(1), axis=0)
            sig_score.fillna(1 / sig_score.shape[1], inplace=True)
            self.sig_score = torch.tensor(sig_score.values)
            sig_genes = list(set(sig_genes))
            adata_omics1 = adata_omics1[:,
                           list(set(adata_omics1.var_names[adata_omics1.var.highly_variable].tolist() + sig_genes))]
        else:
            adata_omics1 = adata_omics1[:, adata_omics1.var.highly_variable]
        # preprocess Protein data
        if issparse(adata_omics2.X):
            adata_omics2.X = adata_omics2.X.toarray()
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.epochs = epochs
        self.zs_dim = zs_dim
        self.zp_dim = zp_dim
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.heads = heads
        self.recon_type_omics1 = recon_type_omics1

        # spatial multi-omics
        self.adata_omics1 = adata_omics1.copy()
        self.adata_omics2 = adata_omics2.copy()
        self.data_omics1 = torch.FloatTensor(
            adata_omics1.X.toarray() if issparse(adata_omics1.X) else adata_omics1.X).to(self.device)
        self.data_omics2 = torch.FloatTensor(
            adata_omics2.X.toarray() if issparse(adata_omics2.X) else adata_omics2.X).to(self.device)
        edge_index_omics1, edge_index_omics2 = adjacent_matrix_preprocessing(adata_omics1, adata_omics2, n_neighbors)
        self.edge_index_omics1 = edge_index_omics1.to(self.device)
        self.edge_index_omics2 = edge_index_omics2.to(self.device)

        # dimension of input feature
        self.dim_input1 = self.data_omics1.shape[1]
        self.dim_input2 = self.data_omics2.shape[1]
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        # weights
        self.weight_omics1 = weight_omics1
        self.weight_omics2 = weight_omics2 * adata_omics1.n_vars / adata_omics2.n_vars
        self.weight_kl = weight_kl
        self.weight_clas = weight_clas

        self.model = mmvaeplus(self.dim_input1, self.dim_input2, zs_dim, zp_dim, hidden_dim1, hidden_dim2, self.device,
                               recon_type_omics1, recon_type_omics2, heads)
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate, weight_decay=weight_decay)

    def train(self, plot_result=False, result_path=None, n_cluster_list=None, test_mode=False):
        if n_cluster_list is None:
            n_cluster_list = [10]
        print("model is executed on: " + str(next(self.model.parameters()).device))
        self.model.train()
        for self.epoch in range(self.epochs):
            self.optimizer.zero_grad()
            losses = self.model.loss(self.data_omics1, self.data_omics2, self.edge_index_omics1, self.edge_index_omics2,
                                     self.label_omics1)
            clas = losses['clas'] * self.weight_clas
            recon_omics1 = (losses["recon_omics1"] + losses["recon_omics1_cross"]) * self.weight_omics1
            recon_omics2 = (losses["recon_omics2"] + losses["recon_omics2_cross"]) * self.weight_omics2
            kl_zs = (losses["kl_zs_omics1"] + losses["kl_zs_omics2"]) * self.weight_kl
            kl_zp = (losses["kl_zp_omics1"] + losses["kl_zp_omics2"]) * self.weight_kl

            loss = recon_omics1 + recon_omics2 + kl_zs + kl_zp + clas
            # loss = recon_omics1 + recon_omics2 + kl_zs + kl_zp + recon_pse_omics1 + dis + clas

            loss.backward()
            clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            if (self.epoch + 1) % 10 == 0 and self.verbose:
                print(
                    f"Epoch: {self.epoch + 1}, recon_omics1: {recon_omics1:.3f}, recon_omics2: {recon_omics2:.3f}, kl_zs: {kl_zs:.3f}, kl_zp: {kl_zp:.3f}, clas: {clas:.3f}")
            if (self.epoch + 1) % 50 == 0 and test_mode:
                data = self.adata_omics1.copy()
                embed = self.encode()
                data.obsm['mmvaeplus'] = F.normalize(torch.tensor(embed), p=2, eps=1e-12, dim=1).detach().cpu().numpy()

                for nc in n_cluster_list:
                    clustering(data, key='mmvaeplus', add_key='mmvaeplus', n_clusters=nc, method='mclust',
                               use_pca=True)
                    prediction = data.obs['mmvaeplus']

                    ari = adjusted_rand_score(data.obs['cluster'], prediction)
                    mi = mutual_info_score(data.obs['cluster'], prediction)
                    nmi = normalized_mutual_info_score(data.obs['cluster'], prediction)
                    ami = adjusted_mutual_info_score(data.obs['cluster'], prediction)
                    hom = homogeneity_score(data.obs['cluster'], prediction)
                    vme = v_measure_score(data.obs['cluster'], prediction)

                    ave_score = (ari + mi + nmi + ami + hom + vme) / 6
                    print("number of clusters: " + str(nc))
                    print("ARI: " + str(ari))
                    print('Average score is: ' + str(ave_score))
                    if plot_result:
                        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                        scanpy.pl.spatial(data, color='ground_truth', ax=ax[0], spot_size=np.sqrt(2), show=False)
                        scanpy.pl.spatial(data, color='mmvaeplus', ax=ax[1], spot_size=np.sqrt(2),
                                          title='mmvaeplus\n ari: ' + str(ari), show=False)
                        plt.tight_layout()
                        plt.show()

                    if result_path is not None:
                        while True:
                            try:
                                result = read_csv(result_path)
                                break
                            except:
                                pass
                        result.loc[len(result.index)] = [self.learning_rate, self.zs_dim, self.zp_dim,
                                                         self.hidden_dim1, self.hidden_dim2, self.weight_omics2,
                                                         self.weight_kl, self.weight_clas, self.recon_type_omics1,
                                                         self.heads, self.epoch + 1, nc, ari, mi, nmi, ami, hom, vme,
                                                         ave_score]
                        result.to_csv(result_path, index=False)
                        print(datetime.now())
                        print(result.tail(1).to_string())
        print("Model training finished!\n")

    def encode(self):
        with torch.no_grad():
            self.model.eval()

            inference_outputs = self.model.inference(self.data_omics1, self.data_omics2, self.edge_index_omics1,
                                                     self.edge_index_omics2)
        embed = torch.cat([(inference_outputs['zs_mu_omics1'] + inference_outputs['zs_mu_omics2']) / 2,
                           inference_outputs['zp_mu_omics1'], inference_outputs['zp_mu_omics2']], dim=-1)
        return F.normalize(embed, p=2, eps=1e-12, dim=1).detach().cpu().numpy()

    def encode_test(self):
        with torch.no_grad():
            self.model.eval()

            inference_outputs = self.model.inference(self.data_omics1, self.data_omics2, self.edge_index_omics1,
                                                     self.edge_index_omics2)
        embed = torch.cat([(inference_outputs['zs_mu_omics1'] + inference_outputs['zs_mu_omics2']) / 2,
                           inference_outputs['zp_mu_omics1'], inference_outputs['zp_mu_omics2']], dim=-1)

        zs_omics1, zp_omics1 = inference_outputs["zs_mu_omics1"], inference_outputs["zp_mu_omics1"]

        z_omics1 = torch.cat([zs_omics1, zp_omics1], dim=-1)
        adata_z_omics1 = AnnData(z_omics1.detach().cpu().numpy())
        adata_z_omics1.obs['label'] = DataFrame(self.label_omics1.cpu()).idxmax(axis=1).values
        adata_z_omics1.obs['label'] = adata_z_omics1.obs['label'].astype('category')
        sc.pp.pca(adata_z_omics1)
        sc.pp.neighbors(adata_z_omics1, 20, metric='cosine')
        sc.tl.umap(adata_z_omics1)
        sc.pl.umap(adata_z_omics1, color=['label'], save=str(self.epoch + 1) + '.pdf')
        return F.normalize(embed, p=2, eps=1e-12, dim=1).detach().cpu().numpy()

    def generation(self):
        with torch.no_grad():
            self.model.eval()
            inference_outputs = self.model.inference(self.data_omics1, self.data_omics2, self.edge_index_omics1,
                                                     self.edge_index_omics2)
            generative_outputs = self.model.generative(inference_outputs)

        data_omics1_hat = NegBinom(generative_outputs["x_omics1_hat"], self.model.theta_omics1.exp()).sample(
            torch.Size([1]))[0, :, :]
        data_omics2_cross_hat = NegBinom(generative_outputs["x_omics2_hat_cross"],
                                         self.model.theta_omics2.exp()).sample(torch.Size([1]))[0, :, :]
        data_omics2_hat = NegBinom(generative_outputs["x_omics2_hat"], self.model.theta_omics2.exp()).sample(
            torch.Size([1]))[0, :, :]
        data_omics1_cross_hat = NegBinom(generative_outputs["x_omics1_hat_cross"],
                                         self.model.theta_omics1.exp()).sample(torch.Size([1]))[0, :, :]
        return {'data_omics1_hat': data_omics1_hat, 'data_omics2_cross_hat': data_omics2_cross_hat,
                'data_omics2_hat': data_omics2_hat, 'data_omics1_cross_hat': data_omics1_cross_hat}
