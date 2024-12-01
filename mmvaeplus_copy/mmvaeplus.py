from datetime import datetime

import numpy as np
import scanpy
import torch
from matplotlib import pyplot as plt
from pandas import read_csv
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
from torch import tensor
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .utils import clustering
from .model import mmvaeplus
from .preprocess import adjacent_matrix_preprocessing


class MMVAEPLUS:
    def __init__(self, adata_omics1, adata_omics2, n_neighbors=20, learning_rate=1e-3, weight_omics1=1, weight_omics2=1,
                 weight_kl=1, heads=1, epochs=600, zs_dim=32, zp_dim=32, hidden_dim1=256, hidden_dim2=64,
                 recon_type_omics1='nb', recon_type_omics2='nb', verbose=True):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.epochs = epochs
        self.zs_dim = zs_dim
        self.zp_dim = zp_dim
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.n_neighbors = n_neighbors
        self.heads = heads
        self.recon_type_omics1 = recon_type_omics1
        self.recon_type_omics2 = recon_type_omics2

        # spatial multi-omics
        self.adata_omics1 = adata_omics1.copy()
        self.adata_omics2 = adata_omics2.copy()
        self.data_omics1 = torch.FloatTensor(
            adata_omics1.X.toarray() if issparse(adata_omics1.X) else adata_omics1.X).to(self.device)
        self.data_omics2 = torch.FloatTensor(
            adata_omics2.X.toarray() if issparse(adata_omics2.X) else adata_omics2.X).to(self.device)

        self.edge_index_omics1, self.edge_index_omics2 = adjacent_matrix_preprocessing(adata_omics1, adata_omics2,
                                                                                       n_neighbors)
        self.edge_index_omics1 = self.edge_index_omics1.to(self.device)
        self.edge_index_omics2 = self.edge_index_omics2.to(self.device)

        # dimension of input feature
        self.dim_input1 = self.data_omics1.shape[1]
        self.dim_input2 = self.data_omics2.shape[1]
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        # weights
        self.weight_omics1 = weight_omics1
        self.weight_omics2 = weight_omics2
        if adata_omics1.n_vars > adata_omics2.n_vars:
            self.weight_omics2 *= adata_omics1.n_vars / adata_omics2.n_vars
        else:
            self.weight_omics1 *= adata_omics2.n_vars / adata_omics1.n_vars
        self.weight_kl = weight_kl

        self.model = mmvaeplus(self.dim_input1, self.dim_input2, zs_dim, zp_dim, hidden_dim1, hidden_dim2,
                               recon_type_omics1, recon_type_omics2, self.device, heads)
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

    def train(self, plot_result=False, result_path=None, dataset=None, n_cluster_list=None, test_mode=False):
        if n_cluster_list is None:
            n_cluster_list = [10]
        print("model is executed on: " + str(next(self.model.parameters()).device))
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            losses = self.model.loss(self.data_omics1, self.data_omics2, self.edge_index_omics1, self.edge_index_omics2)
            recon_omics1 = losses["recon_omics1"] * self.weight_omics1
            recon_omics2 = losses["recon_omics2"] * self.weight_omics2
            kl_zs = losses["kl_zs"] * self.weight_kl
            kl_zp = losses["kl_zp"] * self.weight_kl
            kl_lambd_omics1 = losses["kl_lambd_omics1"] * self.weight_kl
            kl_c_omics1 = losses["kl_c_omics1"] * self.weight_kl
            kl_w_omics1 = losses["kl_w_omics1"] * self.weight_kl
            kl_lambd_omics2 = losses["kl_lambd_omics2"] * self.weight_kl
            kl_c_omics2 = losses["kl_c_omics2"] * self.weight_kl
            kl_w_omics2 = losses["kl_w_omics2"] * self.weight_kl

            loss = recon_omics1 + recon_omics2 + kl_zs + kl_zp + kl_lambd_omics1 + kl_c_omics1 + kl_w_omics1 + kl_lambd_omics2 + kl_c_omics2 + kl_w_omics2

            loss.backward()
            clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                if self.verbose:
                    print(
                        f"Epoch: {epoch + 1}, recon_omics1: {recon_omics1:.4f}, recon_omics2: {recon_omics2:.4f}, kl_zs: {kl_zs:.4f}, kl_zp: {kl_zp:.4f}, kl_lambd_omics1: {kl_lambd_omics1:.4f}, kl_c_omics1: {kl_c_omics1:.4f}, kl_w_omics1: {kl_w_omics1:.4f}, kl_lambd_omics2: {kl_lambd_omics2:.4f}, kl_c_omics2: {kl_c_omics2:.4f}, kl_w_omics2: {kl_w_omics2:.4f}")
                embed = self.encode()
                if (epoch + 1) % 50 == 0 and test_mode:
                    data = self.adata_omics1.copy()
                    if 'cluster' in self.adata_omics2.obs:
                        data.obs['cluster'] = self.adata_omics2.obs['cluster']
                    data.obsm['mmvaeplus'] = F.normalize(torch.tensor(embed), p=2, eps=1e-12,
                                                         dim=1).detach().cpu().numpy()

                    for nc in n_cluster_list:
                        clustering(data, key='mmvaeplus', add_key='mmvaeplus', n_clusters=nc, method='mclust',
                                   use_pca=True if data.obsm['mmvaeplus'].shape[1] > 20 else False)
                        prediction = data.obs['mmvaeplus']
                        if 'cluster' in data.obs:
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
                        else:
                            ari = mi = nmi = ami = hom = vme = ave_score = np.nan
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
                            result.loc[len(result.index)] = [dataset, self.learning_rate, self.zs_dim, self.zp_dim,
                                                             self.hidden_dim1, self.hidden_dim2, self.weight_omics2,
                                                             self.weight_kl, self.n_neighbors, self.recon_type_omics1,
                                                             self.recon_type_omics2, self.heads, epoch + 1, nc, ari, mi,
                                                             nmi, ami, hom, vme, ave_score]
                            result.to_csv(result_path, index=False)
                            print(datetime.now())
                            print(result.tail(1).to_string())
        print("Model training finished!\n")

    def encode(self):
        with torch.no_grad():
            self.model.eval()
            inference_outputs = self.model.inference(self.data_omics1, self.data_omics2, self.edge_index_omics1,
                                                     self.edge_index_omics2)
        embed = F.softplus(torch.cat([(inference_outputs['zs_mu_omics1'] + inference_outputs['zs_mu_omics2']) / 2,
                                      inference_outputs['zp_mu_omics1'], inference_outputs['zp_mu_omics2']], dim=-1))

        return embed.detach().cpu().numpy()

    def encode_test(self):
        with torch.no_grad():
            self.model.eval()
            inference_outputs = self.model.inference(self.data_omics1, self.data_omics2, self.edge_index_omics1,
                                                     self.edge_index_omics2)
        return {
            'zs_omics1': F.normalize(inference_outputs['zs_mu_omics1'], p=2, eps=1e-12, dim=1).detach().cpu().numpy(),
            'zp_omics1': F.normalize(inference_outputs['zp_mu_omics1'], p=2, eps=1e-12, dim=1).detach().cpu().numpy(),
            'zs_omics2': F.normalize(inference_outputs['zs_mu_omics2'], p=2, eps=1e-12, dim=1).detach().cpu().numpy(),
            'zp_omics2': F.normalize(inference_outputs['zp_mu_omics2'], p=2, eps=1e-12, dim=1).detach().cpu().numpy(),
            'zs': F.normalize((inference_outputs['zs_mu_omics1'] + inference_outputs['zs_mu_omics2']) / 2, p=2,
                              eps=1e-12, dim=1).detach().cpu().numpy(), 'z_omics1': F.normalize(
                torch.cat([inference_outputs['zs_mu_omics1'], inference_outputs['zp_mu_omics1']], dim=-1), p=2,
                eps=1e-12, dim=1).detach().cpu().numpy(), 'z_omics2': F.normalize(
                torch.cat([inference_outputs['zs_mu_omics2'], inference_outputs['zp_mu_omics2']], dim=-1), p=2,
                eps=1e-12, dim=1).detach().cpu().numpy(), 'z': F.normalize(torch.cat(
                [(inference_outputs['zs_mu_omics1'] + inference_outputs['zs_mu_omics2']) / 2,
                 inference_outputs['zp_mu_omics1'], inference_outputs['zp_mu_omics2']], dim=-1), p=2, eps=1e-12,
                dim=1).detach().cpu().numpy()}
