'''
Decoder half of DeepCAMA, attempts to reconstruct x from encoded z and m,
and label y
'''

from torch import nn
from src.models.vision_cama.decoder.py import PYMNIST, PYCIFAR, PYIMAGENET
from src.models.vision_cama.decoder.pz import PZMNIST, PZCIFAR, PZIMAGENET
from src.models.vision_cama.decoder.pm import PMMNIST, PMCIFAR, PMIMAGENET
from src.models.vision_cama.decoder.p_merge import PMergeMNIST, PMergeCIFAR, PMergeIMAGENET
import src.utils.config as cfg


class Decoder(nn.Module):
    def __init__(self, dim_y, dim_z, dim_m):
        super(Decoder, self).__init__()

        # Set dimensions of hidden state representations
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.dim_m = dim_m

        if cfg.DATASET in ['mnist', 'emnist_balanced']:
            PY, PZ, PM, PMerge = PYMNIST, PZMNIST, PMMNIST, PMergeMNIST
        elif cfg.DATASET in ['cifar10', 'cifar100']:
            PY, PZ, PM, PMerge = PYCIFAR, PZCIFAR, PMCIFAR, PMergeCIFAR
        elif cfg.DATASET == 'imagenet':
            PY, PZ, PM, PMerge = PYIMAGENET, PZIMAGENET, PMIMAGENET, PMergeIMAGENET
        else:
            raise NotImplementedError(f'Dataset {cfg.DATASET} not supported')

        # Embedding networks
        self.py = PY(dim_y=self.dim_y)
        self.pz = PZ(dim_z=self.dim_z)
        self.pm = PM(dim_m=self.dim_m)
        self.p_merge = PMerge()

    def forward(self, y, z, m):
        # Create hidden representations
        h_y = self.py(y)
        h_z = self.pz(z)
        h_m = self.pm(m)

        # Merge
        h = self.p_merge(h_y, h_z, h_m)

        return h
