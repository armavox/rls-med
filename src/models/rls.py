from functools import partial

import matplotlib.pyplot as plt

import numpy as np
import raster_geometry
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid


class RLSModule(nn.Module):
    """RLS module

    Parameters
    ----------
    input_size : tuple
        Input image size
    hidden_feat : int
        [description]
    out_feat : int
        [description]

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
    """

    def __init__(self, input_size: tuple):
        super().__init__()

        self.img_size = input_size[-2:]  # [H, W]
        in_feat = self.img_size

        self.ug = nn.Parameter(torch.zeros(*in_feat).normal_(std=0.01))
        self.wg = nn.Parameter(torch.zeros(*in_feat).normal_(std=0.01))

        self.uz = nn.Parameter(torch.zeros(*in_feat).normal_(std=0.01))
        self.wz = nn.Parameter(torch.zeros(*in_feat).normal_(std=0.01))
        self.bz = nn.Parameter(torch.zeros(in_feat[0]))

        self.ur = nn.Parameter(torch.zeros(*in_feat).normal_(std=0.01))
        self.wr = nn.Parameter(torch.zeros(*in_feat).normal_(std=0.01))
        self.br = nn.Parameter(torch.zeros(in_feat[0]))

        self.uo = nn.Parameter(torch.zeros(*in_feat).normal_(std=0.01))
        self.wo = nn.Parameter(torch.zeros(*in_feat).normal_(std=0.01))
        self.bo = nn.Parameter(torch.zeros(in_feat[0]))

        self.k1 = nn.Parameter(torch.tensor([2.0]))
        self.k2 = nn.Parameter(torch.tensor([2.0]))

        self.dense = nn.Linear(*in_feat)
        nn.init.xavier_normal_(self.dense.weight)
        self.dense.bias.data.zero_()

        self.writer = None

    def forward(self, input, hidden, writer: SummaryWriter = None, step: int = None):
        """input: image [B, C, H, W], hidden: level set [B, C, H, W]; C == 1"""

        batch_size = input.size(0)

        c1, c2 = self.avg_inside_outside(input, hidden)

        I_c1 = (input - c1) ** 2
        I_c2 = (input - c2) ** 2
        kappa = self.curvature(hidden)
        make_grid_p = partial(make_grid, nrow=4, normalize=True)
        if writer is not None:
            writer.add_image(f"rls/hidden", make_grid_p(hidden.detach()), step)
            writer.add_image(f"rls/I_c1", make_grid_p(I_c1.detach()), step)
            writer.add_image(f"rls/I_c2", make_grid_p(I_c2.detach()), step)
            writer.add_image(f"rls/kappa", make_grid(kappa.detach(), 4), step)

        hidden = self.gru_rls_cell(hidden, kappa, I_c1, I_c2)
        plot(input, 'input')
        plot(hidden, "levelset")
        plot(kappa, "kappa")

        output = self.dense(hidden)
        return (
            output.view(batch_size, *self.img_size),
            hidden.view(batch_size, 1, *self.img_size),
        )

    def curvature(self, hidden):
        """hidden.size: [B, C, H, W], C == 1"""

        hf = F.pad(hidden, pad=[1, 1, 1, 1], mode='replicate')

        dy = (hf[:, :, 2:, 1:-1] - hf[:, :, :-2, 1:-1]) * 0.5
        dx = (hf[:, :, 1:-1, 2:] - hf[:, :, 1:-1, :-2]) * 0.5
        dyy = hf[:, :, 2:, 1:-1] + hf[:, :, :-2, 1:-1] - 2 * hidden
        dxx = hf[:, :, 1:-1, 2:] + hf[:, :, 1:-1, :-2] - 2 * hidden
        dxy = 0.25 * (hf[:, :, 2:, 2:] + hf[:, :, :-2, :-2] - hf[:, :, :-2, 2:] - hf[:, :, 2:, :-2])

        d2x = dx ** 2
        d2y = dy ** 2

        grad2 = d2x + d2y

        t = (dxx * d2y - 2 * dxy * dx * dy + dyy * d2x) / (grad2 * torch.sqrt(grad2) + 1e-8)

        return t

    def avg_inside_outside(self, x, hidden):
        H = hidden
        Hinv = 1. - H
        Hsum = torch.sum(H, dim=[2, 3], keepdim=True)
        Hinvsum = torch.sum(Hinv, dim=[2, 3], keepdim=True)
        Hsum[Hsum == 0] += 1
        Hinvsum[Hinvsum == 0] += 1
        avg_inside = torch.sum(x * H, dim=[2, 3], keepdim=True)
        avg_oustide = torch.sum(x * Hinv, dim=[2, 3], keepdim=True)

        avg_inside /= Hsum
        avg_oustide /= Hinvsum

        return avg_inside, avg_oustide

    def gru_rls_cell(self, hidden, kappa, I_c1, I_c2):
        x = kappa - self.ug @ I_c1 + self.wg @ I_c2
        plot(x, "x")
        z_t = torch.sigmoid(self.uz @ x + self.wz @ hidden + self.bz)
        r_t = torch.sigmoid(self.ur @ x + self.wr @ hidden + self.br)
        o_t = torch.tanh(self.uo @ x + self.wo @ (hidden * r_t) + self.bo)

        return z_t * o_t + (1 - z_t) * hidden


def init_levelset(img_size: tuple, shape: str = "checkerboard"):
    """Levelset init"""

    level_set = torch.zeros(img_size)
    if shape == "checkerboard":
        for i in range(img_size[0]):
            for j in range(img_size[1]):
                level_set[i, j] = np.sin(i * np.pi / 10) * np.sin(j * np.pi / 10)
    elif shape == "circle":
        mask = raster_geometry.circle(img_size, img_size[0] // 2 - 5)
        level_set[mask] = 1
    else:
        raise NotImplementedError
    return level_set


# import matplotlib.pyplot as plt
# plt.imshow((hidden.cpu().detach().numpy()[0][0]))
# plt.colorbar()
# plt.savefig('hidden.png')
# plt.close()


def plot(matrix, name):
    plt.imshow(matrix[0][0].detach().cpu().numpy(), cmap="gray")
    plt.savefig(f"{name}.png")
    plt.close()
