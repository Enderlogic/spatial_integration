from datetime import datetime

import anndata
import numpy as np
import scanpy
import torch
from matplotlib import pyplot as plt
from pandas import read_csv
from scipy.sparse import issparse
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, v_measure_score
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import clip_grad_norm_

from .utils import clustering
from .model import mmvaeplus, NegBinom
from .preprocess import adjacent_matrix_preprocessing


def reinitialize_parameters(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            init.kaiming_normal_(param.data, nonlinearity='relu')
        elif 'bias' in name:
            init.constant_(param.data, 0)


class MMVAEPLUS:
    def __init__(self, adata_omics1, adata_omics2, n_neighbors=20, learning_rate=1e-3, weight_omics1=1, weight_omics2=1,
                 weight_kl=1, weight_var=1, heads=1, epochs=600, zs_dim=32, zp_dim=32,
                 hidden_dim1=256, hidden_dim2=64, verbose=True):
        '''
        :param adata_omics1: spatial omics 1 data
        :param adata_omics2: spatial omics2 data
        :param n_neighbors: number of neighbors in GNN graph
        :param learning_rate: learning rate
        :param weight_omics1: weight of reconstruction loss for omics 1
        :param weight_omics2: weight of reconstruction loss for omics 2
        :param weight_kl: weight of kl loss
        :param recon_type: reconstruction type
        :param heads: number of heads in GNN
        :param epochs: number of epochs
        :param zs_dim: number of shared latent dimensions
        :param zp_dim: number of private latent dimensions
        :param hidden_dim1: number of hidden dimensions in GNN for Omics 1
        :param hidden_dim2: number of hidden dimensions in GNN for Omics 2
        '''
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.epochs = epochs
        self.zs_dim = zs_dim
        self.zp_dim = zp_dim
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.n_neighbors = n_neighbors
        self.heads = heads

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
        self.weight_omics2 = weight_omics2 * adata_omics1.n_vars / adata_omics2.n_vars
        self.weight_kl = weight_kl
        self.weight_var = weight_var

        self.n_restart = 10
        torch.manual_seed(27)
        self.model = mmvaeplus(self.dim_input1, self.dim_input2, zs_dim, zp_dim, hidden_dim1, hidden_dim2, self.device,
                               heads)
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

    def train(self, plot_result=False, result_path=None, dataset=None, n_cluster_list=None, test_mode=False):
        if n_cluster_list is None:
            n_cluster_list = [10]
        print("model is executed on: " + str(next(self.model.parameters()).device))
        self.model.train()
        for epoch in range(self.epochs):
            for param in self.model.measurer_omics1.parameters():
                param.requires_grad = False
            for param in self.model.measurer_omics2.parameters():
                param.requires_grad = False
            self.optimizer.zero_grad()
            losses = self.model.loss(self.data_omics1, self.data_omics2, self.edge_index_omics1, self.edge_index_omics2)
            recon_omics1 = losses["recon_omics1"] * self.weight_omics1
            recon_omics2 = losses["recon_omics2"] * self.weight_omics2
            kl_z = losses["kl_z"] * self.weight_kl
            kl_u = losses["kl_u"] * self.weight_kl
            kl_s = losses["kl_s"] * self.weight_kl
            kl_w = losses["kl_w"] * self.weight_kl
            kl_delta = losses["kl_delta"] * self.weight_kl
            kl_tau = losses["kl_tau"] * self.weight_kl
            kl_lambda = losses["kl_lambda"] * self.weight_kl
            kl_c = losses["kl_c"] * self.weight_kl
            var_omics1_measure = losses["var_omics1_measure"] * self.weight_var
            var_omics2_measure = losses["var_omics2_measure"] * self.weight_var

            loss = recon_omics1 + recon_omics2 + kl_z + kl_u + kl_s + kl_w + kl_delta + kl_tau + kl_lambda + kl_c + var_omics1_measure + var_omics2_measure

            loss.backward()
            clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            if epoch % self.n_restart == 0:
                n_msr = 100
                reinitialize_parameters(self.model.measurer_omics1)
                reinitialize_parameters(self.model.measurer_omics2)
            else:
                n_msr = 5

            for param in self.model.measurer_omics1.parameters():
                param.requires_grad = True
            for param in self.model.measurer_omics2.parameters():
                param.requires_grad = True
            for _ in range(n_msr):
                self.optimizer.zero_grad()
                losses = self.model.measurer_loss(self.data_omics1, self.data_omics2, self.edge_index_omics1,
                                                  self.edge_index_omics2)
                loss = losses["recon_omics1_measure"] * self.weight_omics1 + losses[
                    "recon_omics2_measure"] * self.weight_omics2
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
            if (epoch + 1) % 1 == 0:
                if self.verbose:
                    print(
                        "Epoch: {:d}, recon_omics1: {:.4f}, recon_omics2: {:.4f}, kl_z: {:.4f}, kl_u: {:.4f}, kl_s: {:.4f}, kl_w: {:.4f}, kl_delta: {:.4f}, kl_tau: {:.4f}, kl_lambda: {:.4f}, kl_c: {:.4f}, var_omics1_measure: {:.4f}, var_omics2_measure: {:.4f}".format(
                            epoch + 1, recon_omics1, recon_omics2, kl_z, kl_u, kl_s, kl_w, kl_delta, kl_tau, kl_lambda,
                            kl_c, var_omics1_measure, var_omics2_measure))
                if (epoch + 1) % 50 == 0 and test_mode:
                    embed, _, _ = self.encode()
                    data_omics1 = self.adata_omics1.copy()
                    prediction = embed.argmax(1)
                    data_omics1.obs['mmvaeplus'] = prediction
                    ari = adjusted_rand_score(data_omics1.obs['cluster'], prediction)
                    mi = mutual_info_score(data_omics1.obs['cluster'], prediction)
                    nmi = normalized_mutual_info_score(data_omics1.obs['cluster'], prediction)
                    ami = adjusted_mutual_info_score(data_omics1.obs['cluster'], prediction)
                    hom = homogeneity_score(data_omics1.obs['cluster'], prediction)
                    vme = v_measure_score(data_omics1.obs['cluster'], prediction)

                    ave_score = (ari + mi + nmi + ami + hom + vme) / 6
                    print("ARI: " + str(ari))
                    print('Average score is: ' + str(ave_score))
                    if plot_result:
                        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
                        scanpy.pl.embedding(data_omics1, basis='spatial', color='cluster', ax=ax[0],
                                            title='Ground truth', s=25, show=False)
                        scanpy.pl.embedding(data_omics1, basis='spatial', color='mmvaeplus', ax=ax[1],
                                            title='mmvaeplus\n ari: ' + str(ari), s=25, show=False)
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
                                                         self.weight_kl, self.n_neighbors, self.recon_type, self.heads,
                                                         epoch + 1, ari, mi, nmi, ami, hom, vme, ave_score]
                        result.to_csv(result_path, index=False)
                        print(datetime.now())
                        print(result.tail(1).to_string())
        print("Model training finished!\n")

    def encode(self):
        with torch.no_grad():
            self.model.eval()
            inference_outputs = self.model.inference(self.data_omics1, self.data_omics2, self.edge_index_omics1,
                                                     self.edge_index_omics2)
        embed = F.softmax(torch.cat([inference_outputs['zp_mu_omics1'],
                                     (inference_outputs['zs_mu_omics1'] + inference_outputs['zs_mu_omics2']) / 2,
                                     inference_outputs['zp_mu_omics2']], dim=-1))

        return embed.detach().cpu().numpy(), F.softmax(
            self.model.decoder_omics1.w_m.reshape(-1, self.data_omics1.shape[1]),
            dim=1).detach().cpu().numpy(), F.softmax(
            self.model.decoder_omics2.w_m.reshape(-1, self.data_omics2.shape[1]), dim=1).detach().cpu().numpy()

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

    def generation(self):
        with torch.no_grad():
            self.model.eval()
            inference_outputs = self.model.inference(self.data_omics1, self.data_omics2, self.edge_index_omics1,
                                                     self.edge_index_omics2)
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
