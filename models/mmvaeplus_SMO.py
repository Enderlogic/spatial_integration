# CUB Image-Captions MMVAEplus model specification
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from numpy import sqrt

from .utils import Constants, plot_text_as_image_tensor
from .mmvaeplus import MMVAEplus

from .vae_SMO_modality import SMO_SRT, SMO_SPR



class SMO_srt_spr(MMVAEplus):
    """
    MMVAEplus subclass for CUB Image-Captions Experiment
    """
    def __init__(self, params):
        super(SMO_srt_spr, self).__init__(params, SMO_SRT, SMO_SPR)
        self._ps_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_z), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_z), requires_grad=False)  # logvar
        ])
        self._pp_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False)  # logvar
        ])
        self.vaes[0].llik_scaling = params.llik_scaling_srt

        self.vaes[1].llik_scaling = params.llik_scaling_spr
        self.modelName = 'MMVAEplus_SMO_srt_spr'
        self.params = params

    @property
    def ps_params(self):
        """

        Returns: Parameters of prior distribution for shared latent code

        """
        return self._ps_params[0], F.softplus(self._ps_params[1]) + Constants.eta

    @property
    def pp_params(self):
        """

        Returns: Parameters of prior distribution for modality-specific latent code

        """
        return self._pp_params[0], F.softplus(self._pp_params[1]) + Constants.eta

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


