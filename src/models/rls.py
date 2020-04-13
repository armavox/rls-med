import numpy as np
import raster_geometry
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.nn.init import xavier_normal_, xavier_uniform_

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
        in_feat = np.prod(self.img_size)

        self.U_g = nn.Parameter(torch.zeros(in_feat, in_feat).normal_(std=0.01))
        self.W_g = nn.Parameter(torch.zeros(in_feat, in_feat).normal_(std=0.01))

        self.gru = nn.GRUCell(in_feat, in_feat)
        self.dense = nn.Linear(in_feat, in_feat * 2)

        xavier_uniform_(self.gru.weight_hh)
        xavier_uniform_(self.gru.weight_ih)
        self.gru.bias_ih.data.zero_()
        self.gru.bias_hh.data.zero_()

        xavier_normal_(self.dense.weight)
        self.dense.bias.data.zero_()

    def forward(self, input, hidden):
        """input: image [B, C, H, W], hidden: level set [B, C, H, W]; C == 1"""

        batch_size = input.size(0)

        c1, c2 = self.avg_inside(input, hidden.detach()), self.avg_outside(input, hidden.detach())
        I_c1 = ((input - c1) ** 2).view(batch_size, -1)
        I_c2 = ((input - c2) ** 2).view(batch_size, -1)
        x = (
            self.curvature(hidden).view(batch_size, -1)
            - torch.mm(I_c1, self.U_g)
            + torch.mm(I_c2, self.W_g)
        )

        hidden = self.gru(x, hidden.view(batch_size, -1))
        output = self.dense(hidden)
        return output.view(batch_size, 2, *self.img_size), hidden.view(batch_size, 1, *self.img_size)

    def curvature(self, hidden):
        """hidden.size: [B, C, H, W], C == 1"""

        phi_dx = img_derivative(hidden, SOBEL_X)
        phi_dxx = img_derivative(phi_dx, SOBEL_X)

        phi_dy = img_derivative(hidden, SOBEL_Y)
        phi_dyy = img_derivative(phi_dy, SOBEL_Y)

        phi_dxy = img_derivative(phi_dx, SOBEL_Y)

        kappa = (phi_dxx * phi_dy ** 2 - 2 * phi_dx * phi_dy * phi_dxy + phi_dyy * phi_dx ** 2) / (
            phi_dx ** 2 + phi_dy ** 2
        ) ** (3 / 2)
        kappa[torch.isnan(kappa)] = 0
        return kappa

    def avg_inside(self, input, level_set):
        mask = heaviside(level_set) > 0.5
        inside = input * mask
        return inside.mean()

    def avg_outside(self, input, level_set):
        mask = heaviside(level_set) > 0.5
        outside = input * ~mask
        return outside.mean()


def init_levelset(img_size: tuple, shape: str = "checkerboard"):
    """Levelset init"""

    level_set = torch.zeros(img_size)
    if shape == "checkerboard":
        for i in range(img_size[0]):
            for j in range(img_size[1]):
                level_set[i, j] = np.sin(i * np.pi / 20) * np.sin(j * np.pi / 20)
    elif shape == "circle":
        mask = raster_geometry.circle(img_size, img_size[0] // 2 - 5)
        level_set[mask] = 1
    else:
        raise NotImplementedError
    return level_set


def nth_derivative(f, wrt, n):
    for i in range(n):
        grads = grad(f, wrt, create_graph=True)[0]
        f = grads.sum()
    return grads


def lstm_cell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):

    hx, cx = hidden  # w_ih: (256, 4), b_ih: (256); w_hh: (256, 64), b_hh: (256)
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy

# import matplotlib.pyplot as plt
# plt.imshow((hidden.cpu().detach().numpy()[0][0]))
# plt.colorbar()
# plt.savefig('hidden.png')
# plt.close()
