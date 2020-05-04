from functools import partial

import matplotlib.pyplot as plt

import numpy as np
import raster_geometry
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid

from data.transforms import img_derivative, heaviside, SOBEL_X, SOBEL_Y


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
        nn.init.eye_(self.ug)
        nn.init.eye_(self.wg)

        self.uz = nn.Parameter(torch.zeros(*in_feat).normal_(std=0.01))
        self.wz = nn.Parameter(torch.zeros(*in_feat).normal_(std=0.01))
        self.bz = nn.Parameter(torch.zeros(in_feat[0]))
        nn.init.eye_(self.uz)
        nn.init.eye_(self.wz)

        self.ur = nn.Parameter(torch.zeros(*in_feat).normal_(std=0.01))
        self.wr = nn.Parameter(torch.zeros(*in_feat).normal_(std=0.01))
        self.br = nn.Parameter(torch.zeros(in_feat[0]))
        nn.init.eye_(self.ur)
        nn.init.eye_(self.wr)

        self.uo = nn.Parameter(torch.zeros(*in_feat).normal_(std=0.01))
        self.wo = nn.Parameter(torch.zeros(*in_feat).normal_(std=0.01))
        self.bo = nn.Parameter(torch.zeros(in_feat[0]))
        nn.init.eye_(self.uo)
        nn.init.eye_(self.wo)

        # self.gru = nn.GRUCell(*in_feat)
        self.dense = nn.Linear(*in_feat)

        # nn.init.eye_(self.gru.weight_hh)
        # nn.init.eye_(self.gru.weight_ih)
        # self.gru.bias_ih.data.zero_()
        # self.gru.bias_hh.data.zero_()

        nn.init.xavier_normal_(self.dense.weight)
        self.dense.bias.data.zero_()

        self.writer = None

    def forward(self, input, hidden, writer: SummaryWriter = None, step: int = None):
        """input: image [B, C, H, W], hidden: level set [B, C, H, W]; C == 1"""

        batch_size = input.size(0)

        c1, c2 = self.avg_inside(input, hidden.detach()), self.avg_outside(input, hidden.detach())
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
        # hidden = self.gru(x, hidden.view(batch_size, -1))
        # hidden = F.relu(hidden)
        output = self.dense(hidden)
        return (
            output.view(batch_size, *self.img_size),
            hidden.view(batch_size, 1, *self.img_size),
        )

    def curvature(self, hidden):
        """hidden.size: [B, C, H, W], C == 1"""

        phi_dx = img_derivative(hidden, SOBEL_X)
        phi_dy = img_derivative(hidden, SOBEL_Y)
        phi_dx_n = phi_dx / (phi_dx ** 2 + phi_dy ** 2 + 1e-8) ** 0.5
        phi_dy_n = phi_dy / (phi_dx ** 2 + phi_dy ** 2 + 1e-8) ** 0.5
        return -(torch.abs(phi_dx_n) + torch.abs(phi_dy_n))

    def avg_inside(self, input, level_set):
        mask = heaviside(level_set) > 0.5
        inside = input * mask
        return inside.mean()

    def avg_outside(self, input, level_set):
        mask = heaviside(level_set) > 0.5
        outside = input * ~mask
        return outside.mean()

    def gru_rls_cell(self, hidden, kappa, I_c1, I_c2):

        x = kappa + self.ug @ I_c1 - self.wg @ I_c2
        plot(x, "x")
        z_t = F.sigmoid(self.uz @ x + self.wz @ hidden + self.bz)
        r_t = F.sigmoid(self.ur @ x + self.wr @ hidden + self.br)
        o_t = F.tanh(self.uo @ x + self.wo @ (hidden * r_t) + self.bo)

        return z_t * hidden + (1 - z_t) * o_t


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
    plt.imshow(matrix[0][0], cmap="gray")
    plt.savefig(f"{name}.png")
