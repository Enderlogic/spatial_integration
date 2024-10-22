# Util functions
import math
import os
import shutil
import sys
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment


# Constants
class Constants(object):
    eta = 1e-20


def is_multidata(dataB):
    return isinstance(dataB, list) or isinstance(dataB, tuple)


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}'
              .format(self.name,
                      self.elapsed,
                      time.strftime("%H:%M:%S", self.elapsedH)))


def save_vars(vs, filepath):
    """
    Saves variables to the given filepath in a safe manner.
    """
    if os.path.exists(filepath):
        shutil.copyfile(filepath, '{}.old'.format(filepath))
    torch.save(vs, filepath)


def save_model_light(model, filepath):
    """
    To load a saved model, simply use
    `model.load_state_dict(torch.load('path-to-saved-model'))`.
    """
    save_vars(model.state_dict(), filepath)
    # if hasattr(model, 'vaes'):
        # for vae in model.vaes:
            # fdir, fext = os.path.splitext(filepath)
            # save_vars(vae.state_dict(), fdir + '_' + vae.modelName + fext)

def unpack_data_PM(data, device='cuda'):
    data_nolabel = data[0]
    n_idxs = len(data_nolabel)
    return [data_nolabel[idx].to(device) for idx in range(n_idxs)], data[1].to(device)

def unpack_data_cubIC(data, device='cuda'):
    return [data[0][0].to(device), data[1][0].to(device)]


def get_mean(d, K=100):
    """
    Extract the `mean` parameter for given distribution.
    If attribute not available, estimate from samples.
    """
    try:
        mean = d.mean
    except NotImplementedError:
        samples = d.rsample(torch.Size([K]))
        mean = samples.mean(0)
    return mean


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


class NonLinearLatent_Classifier(nn.Module):
    """ Non-linear Latent classifier defintion. """

    def __init__(self, in_n, out_n):
        super(NonLinearLatent_Classifier, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_n, 64), nn.ReLU(inplace=True),
                                 nn.Linear(64, out_n), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.mlp(x)

def get_10_polymnist_samples(polymnist, num_testing_images, device):
    """
    Function to get PolyMNIST samples for qualitative examples of cross-reconstruction at test time
    """
    samples = []
    for i in range(10):
        while True:
            imgs, target = polymnist.__getitem__(random.randint(0, num_testing_images - 1))
            img_mnist, img_svhn, img_3, img_4, img_5 = imgs
            if target == i:
                img_mnist = img_mnist.to(device)
                img_svhn = img_svhn.to(device)
                img_3 = img_3.to(device)
                img_4 = img_4.to(device)
                img_5 = img_5.to(device)
                # text = text.to(flags.device);
                # samples.append((img_mnist, img_svhn, text, target))
                samples.append((img_mnist, img_svhn, img_3, img_4, img_5, target))
                break
    outputs = []
    for mod in range(5):
        outputs_mod = [samples[digit][mod] for digit in range(10)]
        outputs.append(torch.stack(outputs_mod, dim=0))
    return outputs


class Flatten(torch.nn.Module):
    """
    Helper function for ClfImg class
    """
    def forward(self, x):
        return x.view(x.size(0), -1)


class ClfImg(nn.Module):
    """
    PolyMNIST image-to-digit classifier. Roughly based on the encoder from:
    https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(3, 10, kernel_size=4, stride=2, padding=1),  # -> (10, 14, 14)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=4, stride=2, padding=1),  # -> (20, 7, 7)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            Flatten(),  # -> (980)
            nn.Linear(980, 128),  # -> (128)
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 10)  # -> (10)
        )

    def forward(self, x):
        h = self.encoder(x)
        return F.log_softmax(h, dim=-1)


# Helper function to plot captions so to smoothly log on WandB
def plot_text_as_image_tensor(sentences_lists_of_words, pixel_width=64, pixel_height=384):
    imgs = []
    for sentence in sentences_lists_of_words:
        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        fig = plt.figure(figsize=(pixel_width * px, pixel_height * px))
        plt.text(
            x=1,
            y=0.5,
            s='{}'.format(
                ' '.join(i + '\n' if (n + 1) % 1 == 0
                         else i for n, i in enumerate([word for word in sentence.split() if word != '<eos>']))),
            fontsize=7,
            verticalalignment='center_baseline',
            horizontalalignment='right'
        )
        plt.axis('off')

        # Draw the canvas and retrieve the image as a NumPy array
        fig.canvas.draw()
        image_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_np = image_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Convert the NumPy array to a PyTorch tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255  # Normalize to [0, 1]
        imgs.append(image_tensor)
        # Clean up the figure
        plt.close(fig)
    return torch.stack(imgs, dim=0)




