import torch
from scipy.special import erf
from scipy.linalg import orth
import numpy as np
#
import matplotlib.pyplot as plt


def sample(codeword, basis=None):
    # pseudogaussian = codeword * torch.abs(torch.randn_like(codeword, dtype=torch.float64))
    codeword_np = codeword.numpy()
    rand_np = np.random.randn(*codeword_np.shape)
    pseudogaussian_np = codeword_np * np.abs(rand_np)
    pseudogaussian = torch.from_numpy(pseudogaussian_np).to(dtype=torch.float64)

    ### --- plotting ---
    rand_plt = rand_np.reshape(1, 4, 64, 64)
    psg_plt = pseudogaussian_np.reshape(1, 4, 64, 64)
    diff_plt = np.abs(pseudogaussian_np == rand_np).reshape(1, 4, 64, 64)
    fig, ax = plt.subplots(3, 4)
    for i in range(4):
        im1 = ax[0, i].imshow(rand_plt[0, i], cmap='RdBu', vmin=-4, vmax=4)
        im2 = ax[1, i].imshow(psg_plt[0, i], cmap='RdBu', vmin=-4, vmax=4)
        im3 = ax[2, i].imshow(diff_plt[0, i], cmap='gray', vmin=0, vmax=1)
        if i == 0:
                fig.colorbar(im1, ax=ax[0, i], orientation='vertical', label='init_latents', location='left')
                fig.colorbar(im2, ax=ax[1, i], orientation='vertical', label='pseudogaussian', location='left')
                fig.colorbar(im3, ax=ax[2, i], orientation='vertical', label='black = sign changed', location='left')
        ax[0, i].axis('off')
        ax[1, i].axis('off')
        ax[2, i].axis('off')
    fig.suptitle("PRC Watermarking")
    plt.show()
    ### ---------------
    if basis is None:
        return pseudogaussian
    return pseudogaussian @ basis.T


def recover_posteriors(z, basis=None, variances=None):
    if variances is None:
        default_variance = 1.5
        denominators = np.sqrt(2 * default_variance * (1+default_variance)) * torch.ones_like(z)
    elif type(variances) is float:
        denominators = np.sqrt(2 * variances * (1 + variances))
    else:
        denominators = torch.sqrt(2 * variances * (1 + variances))

    if basis is None:
        return erf(z / denominators)
    else:
        return erf((z @ basis) / denominators)

def random_basis(n):
    gaussian = torch.randn(n, n, dtype=torch.double)
    return orth(gaussian)
