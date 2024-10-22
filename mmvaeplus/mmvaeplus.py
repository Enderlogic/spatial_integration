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
from .model import mmvaeplus, NegBinom
from .preprocess import adjacent_matrix_preprocessing


class MMVAEPLUS:
    def __init__(self, adata_srt, adata_spr, adata_pse=None, n_neighbors=20, learning_rate=1e-3, weight_omics1=1,
                 weight_omics2=1, weight_pse=1, weight_kl=1, weight_dis=1, weight_clas=1, recon_type='nb', heads=1,
                 epochs=600, zs_dim=32, zp_dim=32, hidden_dim1=256, hidden_dim2=64, use_scrna=False, pretrain_ratio=1,
                 verbose=True):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input, Our current model supports 'SPOTS', 'Stereo-CITE-seq', and 'Spatial-ATAC-RNA-seq'. We plan to extend our model for more data types in the future.  
            The default is 'SPOTS'.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2022.    
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight decay to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 1500.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        weight_factors : list, optional
            Weight factors to balance the influcences of different omics data on model training.
    
        Returns
        -------
        The learned representation 'self.emb_combined'.

        '''
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.epochs = epochs
        self.zs_dim = zs_dim
        self.zp_dim = zp_dim
        self.verbose = verbose
        self.use_scrna = use_scrna
        self.learning_rate = learning_rate
        self.pretrain_ratio = pretrain_ratio
        self.n_neighbors = n_neighbors
        self.recon_type = recon_type
        self.heads = heads

        # spatial multi-omics
        self.adata_srt = adata_srt.copy()
        self.adata_spr = adata_spr.copy()
        self.data_srt = torch.FloatTensor(adata_srt.X.toarray() if issparse(adata_srt.X) else adata_srt.X).to(
            self.device)
        self.data_spr = torch.FloatTensor(adata_spr.X.toarray() if issparse(adata_spr.X) else adata_spr.X).to(
            self.device)
        if adata_pse is None:
            self.adata_pse = None
            self.data_pse = None
            self.ctp = None
            self.edge_index_pse = None
            self.edge_index_srt, self.edge_index_spr = adjacent_matrix_preprocessing(adata_srt, adata_spr, n_neighbors)
        else:
            self.edge_index_srt, self.edge_index_spr, self.edge_index_pse = adjacent_matrix_preprocessing(adata_srt,
                                                                                                          adata_spr,
                                                                                                          n_neighbors,
                                                                                                          adata_pse)
            self.adata_pse = adata_pse.copy()
            self.data_pse = torch.FloatTensor(adata_pse.X).to(self.device)
            self.ctp = torch.FloatTensor(adata_pse.obs.values).to(self.device)
            self.edge_index_pse = self.edge_index_pse.to(self.device)
            self.pretrain_epochs = np.floor(epochs * pretrain_ratio)
        self.edge_index_srt = self.edge_index_srt.to(self.device)
        self.edge_index_spr = self.edge_index_spr.to(self.device)

        # dimension of input feature
        self.dim_input1 = self.data_srt.shape[1]
        self.dim_input2 = self.data_spr.shape[1]
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        # weights
        self.weight_omics1 = weight_omics1
        self.weight_omics2 = weight_omics2
        self.weight_pse = weight_pse
        self.weight_kl = weight_kl
        self.weight_dis = weight_dis
        self.weight_clas = weight_clas

        self.predicted_labels = None

        self.model = mmvaeplus(self.dim_input1, self.dim_input2, zs_dim, zp_dim, hidden_dim1, hidden_dim2,
                               self.ctp.shape[1] if adata_pse is not None else 0, self.device, recon_type, heads)
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

    def train(self, plot_result=False, result_path=None, dataset=None, n_cluster_list=None, test_mode=False,
              early_stop=False):
        if n_cluster_list is None:
            n_cluster_list = [10]
        print("model is executed on: " + str(next(self.model.parameters()).device))
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            if self.adata_pse is not None and epoch >= self.pretrain_epochs:
                losses = self.model.loss(self.data_srt, self.data_spr, self.edge_index_srt, self.edge_index_spr,
                                         self.data_pse, self.edge_index_pse, self.ctp)
                recon_pse = losses["recon_pse"] * self.weight_pse
                dis = losses["dis"] * self.weight_dis
                clas = losses['clas'] * self.weight_clas
            else:
                losses = self.model.loss(self.data_srt, self.data_spr, self.edge_index_srt, self.edge_index_spr)
                recon_pse = dis = clas = tensor(0., device=self.device)
            recon_omics1 = losses["recon_omics1"] * self.weight_omics1
            recon_omics1_cross = losses["recon_omics1_cross"] * self.weight_omics1
            recon_omics2 = losses["recon_omics2"] * self.weight_omics2
            recon_omics2_cross = losses["recon_omics2_cross"] * self.weight_omics2
            kl_zs_omics1 = losses["kl_zs_omics1"] * self.weight_kl
            kl_zp_omics1 = losses["kl_zp_omics1"] * self.weight_kl
            kl_zs_omics2 = losses["kl_zs_omics2"] * self.weight_kl
            kl_zp_omics2 = losses["kl_zp_omics2"] * self.weight_kl

            loss = recon_omics1 + recon_omics1_cross + recon_omics2 + recon_omics2_cross + kl_zs_omics1 + kl_zp_omics1 + kl_zs_omics2 + kl_zp_omics2 + recon_pse + dis + clas

            loss.backward()
            clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            # with torch.no_grad():
            #     self.model.eval()
            #     inference_output = self.model.inference(self.data_srt, self.data_spr, self.edge_index_srt,
            #                                             self.edge_index_spr)
            # feature_graph_ori = kneighbors_graph(self.adata_srt.obsm['X_pca'], 20, mode='distance',
            #                                      metric='correlation', include_self=True)
            # feature_graph_updated = kneighbors_graph(
            #     torch.cat([inference_output['zs_mu_omics1'], inference_output['zp_mu_omics1']], 1), 20,
            #     mode='distance', metric='correlation', include_self=True)
            if (epoch + 1) % 10 == 0:
                if self.verbose:
                    print(
                        f"Epoch: {epoch + 1}, recon_omics1: {recon_omics1:.4f}, recon_omics1_cross: {recon_omics1_cross:.4f}, recon_omics2: {recon_omics2:.4f}, recon_omics2_cross: {recon_omics2_cross:.4f}, kl_zs_omics1: {kl_zs_omics1:.4f}, kl_zp_omics1: {kl_zp_omics1:.4f}, kl_zs_omics2: {kl_zs_omics2:.4f}, kl_zp_omics2: {kl_zp_omics2:.4f}, recon_pse: {recon_pse:.4f}, dis: {dis:.4f}, clas: {clas:.4f}")
                embed = self.encode()
                if early_stop:
                    kmeans = KMeans(n_clusters=n_cluster_list[0], random_state=0, n_init='auto').fit_predict(embed)
                    if self.predicted_labels is not None:
                        ari = adjusted_rand_score(self.predicted_labels, kmeans)
                        print(ari)
                    self.predicted_labels = kmeans
                if (epoch + 1) % 50 == 0 and test_mode:
                    data_srt = self.adata_srt.copy()
                    data_srt.obsm['mmvaeplus'] = F.normalize(torch.tensor(embed), p=2, eps=1e-12,
                                                             dim=1).detach().cpu().numpy()

                    for nc in n_cluster_list:
                        clustering(data_srt, key='mmvaeplus', add_key='mmvaeplus', n_clusters=nc, method='mclust',
                                   use_pca=True)
                        prediction = data_srt.obs['mmvaeplus']

                        ari = adjusted_rand_score(data_srt.obs['cluster'], prediction)
                        mi = mutual_info_score(data_srt.obs['cluster'], prediction)
                        nmi = normalized_mutual_info_score(data_srt.obs['cluster'], prediction)
                        ami = adjusted_mutual_info_score(data_srt.obs['cluster'], prediction)
                        hom = homogeneity_score(data_srt.obs['cluster'], prediction)
                        vme = v_measure_score(data_srt.obs['cluster'], prediction)

                        ave_score = (ari + mi + nmi + ami + hom + vme) / 6
                        print("number of clusters: " + str(nc))
                        print("ARI: " + str(ari))
                        print('Average score is: ' + str(ave_score))
                        if plot_result:
                            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                            scanpy.pl.spatial(data_srt, color='ground_truth', ax=ax[0], spot_size=np.sqrt(2),
                                              show=False)
                            scanpy.pl.spatial(data_srt, color='spaint', ax=ax[1], spot_size=np.sqrt(2),
                                              title='spaint\n ari: ' + str(ari), show=False)
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
                                                             self.weight_kl, self.n_neighbors, self.recon_type,
                                                             self.heads, epoch + 1, nc, ari, mi, nmi, ami, hom, vme,
                                                             ave_score]
                            result.to_csv(result_path, index=False)
                            print(datetime.now())
                            print(result.tail(1).to_string())
        print("Model training finished!\n")

    def encode(self):
        with torch.no_grad():
            self.model.eval()
            inference_outputs = self.model.inference(self.data_srt, self.data_spr, self.edge_index_srt,
                                                     self.edge_index_spr)
        embed = torch.cat([(inference_outputs['zs_mu_omics1'] + inference_outputs['zs_mu_omics2']) / 2,
                           inference_outputs['zp_mu_omics1'], inference_outputs['zp_mu_omics2']], dim=-1)

        return F.normalize(embed, p=2, eps=1e-12, dim=1).detach().cpu().numpy()

    def encode_test(self):
        with torch.no_grad():
            self.model.eval()
            inference_outputs = self.model.inference(self.data_srt, self.data_spr, self.edge_index_srt,
                                                     self.edge_index_spr)
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

    def generation(self):
        with torch.no_grad():
            self.model.eval()
            inference_outputs = self.model.inference(self.data_srt, self.data_spr, self.edge_index_srt,
                                                     self.edge_index_spr)
            generative_outputs = self.model.generative(inference_outputs)

        data_srt_hat = NegBinom(generative_outputs["x_omics1_hat"], self.model.theta_omics1.exp()).sample(
            torch.Size([1]))[0, :, :]
        data_spr_cross_hat = NegBinom(generative_outputs["x_omics2_hat_cross"], self.model.theta_omics2.exp()).sample(
            torch.Size([1]))[0, :, :]
        data_spr_hat = NegBinom(generative_outputs["x_omics2_hat"], self.model.theta_omics2.exp()).sample(
            torch.Size([1]))[0, :, :]
        data_srt_cross_hat = NegBinom(generative_outputs["x_omics1_hat_cross"], self.model.theta_omics1.exp()).sample(
            torch.Size([1]))[0, :, :]
        return {'data_srt_hat': data_srt_hat, 'data_spr_cross_hat': data_spr_cross_hat, 'data_spr_hat': data_spr_hat,
                'data_srt_cross_hat': data_srt_cross_hat}
