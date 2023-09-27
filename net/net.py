import torch
import torch.nn as nn
from mri_tools import AtA

class Dw(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dw, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x):
        return x + self.layers(x)


class ConjugatedGrad(nn.Module):
    def __init__(self):
        super(ConjugatedGrad, self).__init__()

    def forward(self, rhs, csm, mask, lam):
        rhs = torch.view_as_complex(rhs.permute(0, 2, 3, 1).contiguous())
        x = torch.zeros_like(rhs)
        i, r, p = 0, rhs, rhs
        rTr = torch.sum(torch.conj(r) * r, dim=(-2, -1)).real
        num_iter, epsilon = 10, 1e-10
        for i in range(num_iter):
            Ap = AtA(p, csm, mask) + lam * p
            alpha = rTr / torch.sum(torch.conj(p) * Ap, dim=(-2, -1)).real
            x = x + alpha[:, None, None] * p
            r = r - alpha[:, None, None] * Ap
            rTrNew = torch.sum(torch.conj(r) * r, dim=(-2, -1)).real
            if rTrNew.max() < epsilon:
                break
            beta = rTrNew / rTr
            rTr = rTrNew
            p = r + beta[:, None, None] * p
        return x

class MLP_FastMRI(nn.Module):
    def __init__(self):
        super(MLP_FastMRI, self).__init__()
        self.fc1 = nn.Linear(320*320, 1024)

    def forward(self, x):
        x = x.view(-1, 320*320)
        x = self.fc1(x)
        x = F.relu(x)
        return x

class MLP_t1(nn.Module):
    def __init__(self):
        super(MLP_t1, self).__init__()
        self.fc1 = nn.Linear(240*240, 1024)

    def forward(self, x):
        x = x.view(-1, 240*240)
        x = self.fc1(x)
        x = F.relu(x)
        return x

class MoDL(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(MoDL, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.layers = Dw(self.in_channels, self.out_channels)
        self.lam = nn.Parameter(torch.FloatTensor([0.05]).to(tuple(self.layers.parameters())[0].device), requires_grad=True)
        self.CG = ConjugatedGrad()

    def forward(self, under_img, csm, under_mask):
        x = under_img
        for i in range(self.num_layers):
            x = self.layers(x)
            x = under_img + self.lam * x
            x = self.CG(x, csm, under_mask, self.lam)
            x = torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()
        x_final = x
        return x_final

class ParallelNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, mlp_config):
        super(ParallelNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.up_network = MoDL(self.in_channels, self.out_channels, self.num_layers)
        self.down_network = MoDL(self.in_channels, self.out_channels, self.num_layers)
        self.mlp=globals()[mlp_config.type](**mlp_config.args)

    def forward(self, under_image_up, mask_up, under_image_down, mask_down, csm):

        output_up = self.up_network(under_image_up, csm, mask_up)
        output_down = self.down_network(under_image_down, csm, mask_down)

        return output_up, output_down