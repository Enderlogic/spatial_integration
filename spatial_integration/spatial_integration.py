from datetime import datetime

import torch
from torch import tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .model import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing


class spatial_integration_train:
    def __init__(self, data, datatype='SPOTS', random_seed=2022, learning_rate=0.0001, weight_decay=0.00, epochs=600,
                 dim_input=3000, dim_output=64, dropout=.05, weight_factors=[1, 5, 1, 1, 0, 0, 0]):
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
        self.data = data.copy()
        self.datatype = datatype
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dropout = dropout
        self.weight_factors = weight_factors

        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)

        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs

        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

        if data['adata_pse_srt'] is not None:
            self.num_cell_type = data['adata_pse_srt'].num_cell_type
            self.adata_pse_srt = DataLoader(self.data['adata_pse_srt'], batch_size=1, shuffle=True)
        else:
            self.num_cell_type = 0
            self.adata_pse_srt = None

        if self.datatype == 'SPOTS':
            self.epochs = 600
            self.weight_factors = [1, 5, 1, 1, 0, 0, 0]
        elif self.datatype == 'Stereo-CITE-seq':
            self.epochs = 1500
            self.weight_factors = [1, 10, 1, 10, 0, 0, 0]
        elif self.datatype == '10x':
            self.epochs = 200
            self.weight_factors = [1, 5, 1, 10, 0, 0, 0]
        elif self.datatype == 'Spatial-epigenome-transcriptome':
            self.epochs = 1600
            self.weight_factors = [1, 5, 1, 1, 0, 0, 0]

    def train(self):
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2,
                                     self.num_cell_type, self.dropout).to(self.device)
        print("model is executed on: " + str(next(self.model.parameters()).device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        self.model.train()
        for epoch in range(self.epochs):
            # original training process in spatialglue
            self.model.train()
            self.optimizer.zero_grad()
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1,
                                 self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)

            # reconstruction loss
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])

            # correspondence loss
            self.loss_corr_omics1 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_across_recon'])
            self.loss_corr_omics2 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_across_recon'])

            if self.adata_pse_srt is not None:
                # discrimination loss
                dis = self.model.dis(results['emb_latent_omics1'])
                self.loss_dis_omics1 = F.cross_entropy(dis, tensor([1] * self.n_cell_omics1, device=self.device))
            else:
                self.loss_dis_omics1 = tensor(0., device=self.device)
            loss = self.weight_factors[0] * self.loss_recon_omics1 + self.weight_factors[1] * self.loss_recon_omics2 + \
                   self.weight_factors[2] * self.loss_corr_omics1 + self.weight_factors[
                       3] * self.loss_corr_omics2 + self.loss_dis_omics1

            loss.backward()
            self.optimizer.step()

            # additional training process to integrate information from scRNA-seq data into model
            if self.adata_pse_srt is not None:
                loss_recon_pse_srt = 0
                loss_clas = 0
                loss_dis_pse_srt = 0
                for feat, y, ex_adj in self.adata_pse_srt:
                    self.optimizer.zero_grad()
                    feat, y, ex_adj = feat.squeeze().to(self.device), y.squeeze().to(self.device), ex_adj.squeeze().to(
                        self.device)
                    emb_latent_pse_srt = self.model.encoder_omics1(feat, ex_adj)

                    # reconstruction loss
                    emb_recon_pse_srt = self.model.decoder_omics1(emb_latent_pse_srt, ex_adj)
                    self.loss_recon_pse_srt = F.mse_loss(feat, emb_recon_pse_srt)

                    # classification loss
                    decon = self.model.decon_model(emb_latent_pse_srt)
                    self.loss_clas = F.cross_entropy(decon, y)

                    # discrimination loss
                    dis = self.model.dis(emb_latent_pse_srt)
                    self.loss_dis_pse_srt = F.cross_entropy(dis, tensor([0] * feat.shape[0], device=self.device))
                    loss = self.weight_factors[4] * self.loss_recon_pse_srt + self.weight_factors[5] * self.loss_clas + \
                           self.weight_factors[6] * self.loss_dis_pse_srt
                    loss.backward()
                    self.optimizer.step()
                    loss_recon_pse_srt += self.loss_recon_pse_srt
                    loss_clas += self.loss_clas
                    loss_dis_pse_srt += self.loss_dis_pse_srt
            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch}, recon_omics1: {self.loss_recon_omics1:.4f}, recon_omics2: {self.loss_recon_omics2:.4f}, corr_omics1: {self.loss_corr_omics1:.4f}, corr_omics2: {self.loss_corr_omics2:.4f}")
                if self.adata_pse_srt is not None:
                    print(
                        f"recon_pse_srt: {loss_recon_pse_srt / len(self.adata_pse_srt):.4f}, clas: {loss_clas / len(self.adata_pse_srt):.4f}, dis_omics1: {self.loss_dis_omics1:.4f}, dis_pse_srt: {loss_dis_pse_srt / len(self.adata_pse_srt):.4f}")

        print("Model training finished!\n")

        with torch.no_grad():
            self.model.eval()
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1,
                                 self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)

        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'spatial_integration': emb_combined.detach().cpu().numpy(),
                  'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
                  'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),
                  'alpha': results['alpha'].detach().cpu().numpy()}

        return output
