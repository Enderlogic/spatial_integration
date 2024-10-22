# CUB Image-Captions MMVAEplus model specification
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from numpy import sqrt, prod
import torch.distributions as dist
from torchvision.utils import save_image, make_grid
from utils import Constants, plot_text_as_image_tensor
from .mmvaeplus import MMVAEplus
from .vae_CUB_image_modality import CUB_Image
from .vae_CUB_captions_modality import CUB_Sentence
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset
from dataset_CUB import resampler


# Constants
maxSentLen = 32
minOccur = 3


class CUB_Image_Captions(MMVAEplus):
    """
    MMVAEplus subclass for CUB Image-Captions Experiment
    """
    def __init__(self, params):
        super(CUB_Image_Captions, self).__init__(dist.Normal if params.priorposterior == 'Normal' else dist.Laplace, params, CUB_Image, CUB_Sentence)
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_z), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_z), requires_grad=False)  # logvar
        ])
        self._pw_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False)  # logvar
        ])
        self.vaes[0].llik_scaling = self.vaes[1].maxSentLen / prod(self.vaes[0].dataSize) #\
        #     if params.llik_scaling == 0 else params.llik_scaling
        self.vaes[1].llik_scaling = params.llik_scaling_sent
        self.modelName = 'MMVAEplus_CUB'
        self.params = params

    @property
    def pz_params(self):
        """

        Returns: Parameters of prior distribution for shared latent code

        """
        if self.params.priorposterior == 'Normal':
            return self._pz_params[0], F.softplus(self._pz_params[1]) + Constants.eta
        else:
            return self._pz_params[0], F.softmax(self._pz_params[1], dim=-1) * self._pz_params[1].size(-1) + Constants.eta

    @property
    def pw_params(self):
        """

        Returns: Parameters of prior distribution for modality-specific latent code

        """
        if self.params.priorposterior == 'Normal':
            return self._pw_params[0], F.softplus(self._pw_params[1]) + Constants.eta
        else:
            return self._pw_params[0], F.softmax(self._pw_params[1], dim=-1) * self._pw_params[1].size(-1) + Constants.eta

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        # load base datasets
        t1, s1 = self.vaes[0].getDataLoaders(batch_size, shuffle, device)
        t2, s2 = self.vaes[1].getDataLoaders(batch_size, shuffle, device)

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train_loader = DataLoader(TensorDataset([
            ResampleDataset(t1.dataset, resampler, size=len(t1.dataset) * 10),
            t2.dataset]), batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(TensorDataset([
            ResampleDataset(s1.dataset, resampler, size=len(s1.dataset) * 10),
            s2.dataset]), batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader, test_loader

    def self_and_cross_modal_generation(self, data, num=10,N=10):
        """
        Self- and cross-modal generation.
        Args:
            data: Input
            num: Number of samples to be considered for generation
            N: Number of generations

        Returns:
            Generations
        """
        recon_triess = [[[] for i in range(num)] for j in range(num)]
        outputss = [[[] for i in range(num)] for j in range(num)]
        for i in range(N):
            recons_mat = super(CUB_Image_Captions, self).self_and_cross_modal_generation([d[:num] for d in data])
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                    if o == 0:
                        recon = recon.squeeze(0).cpu()
                        recon_triess[r][o].append(recon)
                    else:
                        if i < 3: # Lower number of generations for cross-modal text generation
                            recon = recon.squeeze(0).cpu()
                            recon_triess[r][o].append(self._plot_sentences_as_tensor(recon))
        for r, recons_list in enumerate(recons_mat):
            if r == 1:
                input_data = self._plot_sentences_as_tensor(data[r][:num]).cpu()
            else:
                input_data = data[r][:num].cpu()

            for o, recon in enumerate(recons_list):
                outputss[r][o] = make_grid(torch.cat([input_data]+recon_triess[r][o], dim=2), nrow=num)
        return outputss

    def self_and_cross_modal_generation_for_fid_calculation(self, data, savePath, i):
        """
        Self- and cross-modal reconstruction for FID calculation.
        Args:
            data: Input
            savePath: Save path
            i: image index for naming
        """
        recons_mat = super(CUB_Image_Captions, self).self_and_cross_modal_generation([d for d in data])
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                if o == 0:
                    recon = recon.squeeze(0).cpu()

                    for image in range(recon.size(0)):
                        save_image(recon[image, :, :, :],
                                    '{}/m{}/m{}/{}_{}.png'.format(savePath, r,o, image, i))

    def save_test_samples_for_fid_calculation(self, data, savePath, i):
        """
        Save test samples for FID calculation.
        Args:
            data: Input
            savePath: Save path
            i: image index for naming
        """
        o = 0
        imgs = data[0].cpu()
        for image in range(imgs.size(0)):
            save_image(imgs[image, :, :, :], '{}/test/m{}/{}_{}.png'.format(savePath, o, image, i))

    def _plot_sentences_as_tensor(self, batched_text_modality):
        """

        Args:
            batched_text_modality: Gets batch of text modality (as it is in input to forward function)

        Returns:
            Tensor with a corresponding plot containing the given text
        """
        sentences_processed = self._sent_process(batched_text_modality.argmax(-1))
        sentences_worded = [' '.join(self.vaes[1].i2w[str(word)] for word in sent if self.vaes[1].i2w[str(word)] != '<pad>') for sent in sentences_processed]
        return plot_text_as_image_tensor(sentences_worded, pixel_width=64, pixel_height=384)

    def generate_unconditional(self, N=100,coherence_calculation=False, fid_calculation=False, savePath=None, tranche=None):
        """
        Generate unconditional
        Args:
            N: Number of generations.
            coherence_calculation: Whether it serves for coherence calculation
            fid_calculation: Whether it serves for fid calculation
            savePath: additional argument for fid calculation save path for images
            tranche: argument needed for naming of images when saved for fid calculation

        Returns:

        """
        outputs = []
        samples_list = super(CUB_Image_Captions, self).generate_unconditional(N)
        if coherence_calculation:
            return [samples.data.cpu() for samples in samples_list]
        elif fid_calculation:
            for i, samples in enumerate(samples_list):
                if i == 0:
                    # Only first modality (images) is considered for FID calculation
                    samples = samples.data.cpu()
                    # wrangle things so they come out tiled
                    # samples = samples.view(N, *samples.size()[1:])
                    for image in range(samples.size(0)):
                        save_image(samples[image, :, :, :], '{}/random/m{}/{}_{}.png'.format(savePath, i, tranche, image))
                else:
                    continue
        else:
            for i, samples in enumerate(samples_list):
                if i == 0:
                    # Image modality
                    samples = samples.data.cpu()
                    samples = samples.view(samples.size()[0], *samples.size()[1:])
                    outputs.append(make_grid(samples, nrow=int(sqrt(N))))
                else:
                    # Text modality
                    samples = samples.data.cpu()
                    samples = samples.view(samples.size()[0], *samples.size()[1:])
                    outputs.append(make_grid(self._plot_sentences_as_tensor(samples), nrow=int(sqrt(N))))

        return outputs

    def _sent_process(self, sentences):
        return [self.vaes[1].fn_trun(self.vaes[1].fn_2i(s)) for s in sentences]


