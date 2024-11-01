import itertools
import os
from datetime import datetime

import numpy as np
import pandas
import scanpy
import torch
from anndata import read_h5ad, AnnData
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame, Series
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
from torch import tensor
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import scanpy as sc

from .utils import clustering
from .model import mmvaeplus, NegBinom
from .preprocess import adjacent_matrix_preprocessing, pse_srt_from_scrna, ST_preprocess, clr_normalize_each_cell


class MMVAEPLUS:
    def __init__(self, adata_omics1, adata_omics2, adata_sc_omics1=None, n_batches=0, n_neighbors=20,
                 learning_rate=1e-3, weight_decay=0.00, weight_omics1=1, weight_omics2=1, weight_pse_omics1=1,
                 weight_kl=1, weight_dis=1, weight_clas=1, recon_type_omics1='nb', recon_type_omics2='binomial',
                 heads=1, epochs=600, zs_dim=32, zp_dim=32, hidden_dim1=256, hidden_dim2=64, use_scrna=False,
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

        # preprocss ST data
        self.epoch = 0
        if adata_sc_omics1 is not None:
            common_genes = [g for g in adata_omics1.var_names if g in adata_sc_omics1.var_names]
            adata_omics1 = adata_omics1[:, common_genes]
            adata_sc_omics1 = adata_sc_omics1[:, common_genes]
        adata_omics1 = ST_preprocess(adata_omics1, n_top_genes=3000, n_comps=50)
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
        self.use_scrna = use_scrna
        self.learning_rate = learning_rate
        self.heads = heads
        self.n_batches = n_batches

        # spatial multi-omics
        self.adata_omics1 = adata_omics1.copy()
        self.adata_omics2 = adata_omics2.copy()
        self.data_omics1 = torch.FloatTensor(
            adata_omics1.X.toarray() if issparse(adata_omics1.X) else adata_omics1.X).to(self.device)
        self.data_omics2 = torch.FloatTensor(
            adata_omics2.X.toarray() if issparse(adata_omics2.X) else adata_omics2.X).to(self.device)

        if adata_sc_omics1 is not None and n_batches > 0:
            adata_pse_omics1 = []
            lam_list = [4, 6, 8, 10]
            max_cell_types_in_spot_list = [2, 4, 6, 8]
            param_list = list(itertools.product(*[lam_list, max_cell_types_in_spot_list]))
            spot_num = adata_omics1.n_obs
            for i in range(n_batches):
                lam, max_cell_types_in_spot = param_list[i]
                filename = 'Dataset/human_lymph_node/adata_pse_omics1_' + str(lam) + '_' + str(
                    max_cell_types_in_spot) + '_' + str(spot_num) + '.h5ad'
                if os.path.exists(filename):
                    temp = read_h5ad(filename)
                else:
                    temp = pse_srt_from_scrna(adata_sc_omics1, spot_num=spot_num, lam=lam,
                                              max_cell_types_in_spot=max_cell_types_in_spot)
                    temp = ST_preprocess(temp, highly_variable_genes=False, n_comps=50)
                    temp = temp[:, adata_omics1.var_names]
                    temp.obs = temp.obs[adata_sc_omics1.obs['celltype'].unique()]
                    temp.write_h5ad(filename)
                adata_pse_omics1.append(temp)
        else:
            adata_pse_omics1 = None
        edge_index_omics1, edge_index_omics2, edge_index_pse_omics1 = adjacent_matrix_preprocessing(adata_omics1,
                                                                                                    adata_omics2,
                                                                                                    adata_pse_omics1,
                                                                                                    n_neighbors)
        self.edge_index_omics1 = edge_index_omics1.to(self.device)
        self.edge_index_omics2 = edge_index_omics2.to(self.device)

        if adata_sc_omics1 is None or n_batches == 0:
            self.adata_pse_omics1 = self.data_pse_omics1 = self.ctp = self.edge_index_pse_omics1 = None
            self.label_dim = 0
        else:
            self.adata_pse_omics1 = [temp.copy() for temp in adata_pse_omics1]
            self.data_pse_omics1 = [torch.FloatTensor(temp.X).to(self.device) for temp in adata_pse_omics1]
            self.ctp = [torch.FloatTensor(temp.obs.values).to(self.device) for temp in self.adata_pse_omics1]
            self.edge_index_pse_omics1 = [temp.to(self.device) for temp in edge_index_pse_omics1]
            self.label_dim = adata_sc_omics1.obs['celltype'].nunique()

        # dimension of input feature
        self.dim_input1 = self.data_omics1.shape[1]
        self.dim_input2 = self.data_omics2.shape[1]
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        # weights
        self.weight_omics1 = weight_omics1
        self.weight_omics2 = weight_omics2 * adata_omics1.n_vars / adata_omics2.n_vars
        self.weight_pse_omics1 = weight_pse_omics1
        self.weight_kl = weight_kl
        self.weight_clas = weight_clas
        self.weight_dis = weight_dis

        self.model = mmvaeplus(self.dim_input1, self.dim_input2, zs_dim, zp_dim, hidden_dim1, hidden_dim2,
                               self.label_dim, self.device, recon_type_omics1, recon_type_omics2, heads,
                               n_batches + 1 if adata_sc_omics1 is not None else 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate, weight_decay=weight_decay)

    def train(self, plot_result=False, result_path=None, dataset=None, n_cluster_list=None, test_mode=False):
        if n_cluster_list is None:
            n_cluster_list = [10]
        print("model is executed on: " + str(next(self.model.parameters()).device))
        self.model.train()
        for self.epoch in range(self.epochs):
            self.optimizer.zero_grad()
            losses = self.model.loss(self.data_omics1, self.data_omics2, self.edge_index_omics1, self.edge_index_omics2,
                                     self.data_pse_omics1, self.edge_index_pse_omics1, self.ctp)

            recon_pse_omics1 = losses["recon_pse_omics1"] * self.weight_pse_omics1
            dis = losses["dis"] * self.weight_dis
            clas = losses['clas'] * self.weight_clas
            recon_omics1 = (losses["recon_omics1"] + losses["recon_omics1_cross"]) * self.weight_omics1
            recon_omics2 = (losses["recon_omics2"] + losses["recon_omics2_cross"]) * self.weight_omics2
            kl_zs = (losses["kl_zs_omics1"] + losses["kl_zs_omics2"]) * self.weight_kl
            kl_zp = (losses["kl_zp_omics1"] + losses["kl_zp_omics2"]) * self.weight_kl
            kl_zs_pse_omics1 = losses["kl_zs_pse_omics1"] * self.weight_kl
            kl_zp_pse_omics1 = losses["kl_zp_pse_omics1"] * self.weight_kl

            loss = recon_omics1 + recon_omics2 + kl_zs + kl_zp + recon_pse_omics1 + dis + clas + kl_zs_pse_omics1 + kl_zp_pse_omics1
            # loss = recon_omics1 + recon_omics2 + kl_zs + kl_zp + recon_pse_omics1 + dis + clas

            loss.backward()
            clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            if (self.epoch + 1) % 10 == 0 and self.verbose:
                print(
                    f"Epoch: {self.epoch + 1}, recon_omics1: {recon_omics1:.3f}, recon_omics2: {recon_omics2:.3f}, kl_zs: {kl_zs:.3f}, kl_zp: {kl_zp:.3f}, recon_pse_omics1: {recon_pse_omics1:.3f}, kl_zs_pse_omics1: {kl_zs_pse_omics1:.3f}, kl_zp_pse_omics1: {kl_zp_pse_omics1:.3f}, dis: {dis:.3f}, clas: {clas:.3f}")
            if (self.epoch + 1) % 10 == 0 and test_mode:
                data = self.adata_omics1.copy()
                embed = self.encode_test(use_pse_omics1=True if self.n_batches > 0 else False)
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
                        result.loc[len(result.index)] = [dataset, self.learning_rate, self.zs_dim, self.zp_dim,
                                                         self.hidden_dim1, self.hidden_dim2, self.heads,
                                                         self.weight_omics1, self.weight_omics2, self.weight_kl,
                                                         self.weight_pse_omics1, self.weight_dis, self.weight_clas,
                                                         self.n_batches, self.epoch + 1, nc, ari, mi, nmi, ami, hom,
                                                         vme, ave_score]
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

    def encode_test(self, use_pse_omics1=False):
        with torch.no_grad():
            self.model.eval()

            inference_outputs = self.model.inference(self.data_omics1, self.data_omics2, self.edge_index_omics1,
                                                     self.edge_index_omics2,
                                                     self.data_pse_omics1 if use_pse_omics1 else None,
                                                     self.edge_index_pse_omics1 if use_pse_omics1 else None)
        embed = torch.cat([(inference_outputs['zs_mu_omics1'] + inference_outputs['zs_mu_omics2']) / 2,
                           inference_outputs['zp_mu_omics1'], inference_outputs['zp_mu_omics2']], dim=-1)
        if use_pse_omics1:
            zs_omics1_combined = torch.cat(
                [inference_outputs["zs_mu_omics1"], torch.cat(inference_outputs["zs_pse_omics1"])])
            zp_omics1_combined = torch.cat(
                [inference_outputs["zp_mu_omics1"], torch.cat(inference_outputs["zp_pse_omics1"])])
        else:
            zs_omics1_combined, zp_omics1_combined = inference_outputs["zs_mu_omics1"], inference_outputs[
                "zp_mu_omics1"]

        z_omics1_combined = torch.cat([zs_omics1_combined, zp_omics1_combined], dim=-1)
        adata_z_omics1_combined = AnnData(z_omics1_combined.detach().cpu().numpy())
        adata_z_omics1_combined.obs['batch'] = np.array(range(self.n_batches + 1)).repeat(embed.shape[0])
        adata_z_omics1_combined.obs['batch'] = adata_z_omics1_combined.obs['batch'].astype('category')
        ct = []
        ct.append(Series([self.label_dim]).repeat(embed.shape[0]))
        for i in range(self.n_batches):
            ct.append(DataFrame(self.ctp[i].cpu()).idxmax(axis=1))
        ct = pandas.concat(ct)
        adata_z_omics1_combined.obs['label'] = ct.values
        adata_z_omics1_combined.obs['label'] = adata_z_omics1_combined.obs['label'].astype('category')
        sc.pp.pca(adata_z_omics1_combined)
        sc.pp.neighbors(adata_z_omics1_combined, 20, metric='cosine')
        sc.tl.umap(adata_z_omics1_combined)
        sc.pl.umap(adata_z_omics1_combined, color=['batch', 'label'], save=str(self.epoch + 1) + '.pdf')
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
