import torch
import torch.nn as nn

from src.models.vision_cama.encoder.qm import QMMNIST, QMCIFAR, QMIMAGENET
from src.models.vision_cama.encoder.qz import QZMNIST, QZCIFAR, QZIMAGENET

import src.utils.config as cfg


class VariationalEncoder(nn.Module):
    def __init__(self, dim_y, dim_z, dim_m):
        super(VariationalEncoder, self).__init__()

        # Set dimensions of hidden state representations
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.dim_m = dim_m

        # Instantiate encoder distributions
        if cfg.DATASET in ['mnist', 'emnist_balanced']:
            QM, QZ = QMMNIST, QZMNIST
        elif cfg.DATASET in ['cifar10', 'cifar100']:
            QM, QZ = QMCIFAR, QZCIFAR
        elif cfg.DATASET == 'imagenet':
            QM, QZ = QMIMAGENET, QZIMAGENET
        else:
            raise NotImplementedError(f'Dataset {cfg.DATASET} not supported')

        self.qm = QM(dim_m=self.dim_m)
        self.qz = QZ(dim_y=self.dim_y, dim_z=self.dim_z, dim_m=self.dim_m)

    def forward(self, x, y, infer_m=False):
        if infer_m:
            # Sample m ~ q(m|x) unknown
            m = self.qm(x)
        else:
            # Otherwise claim clean sample and set to zero
            m = torch.zeros((x.shape[0], self.dim_m)).to(cfg.DEVICE)

        # Sample z ~ q(z|x,y,m)
        z = self.qz(x, y, m)

        return m, z
