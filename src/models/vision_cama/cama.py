import torch
from torch import nn
import torch.nn.functional as F

from src.models.vision_cama.decoder.decoder import Decoder
from src.models.vision_cama.encoder.variational_encoder import VariationalEncoder
import src.utils.config as cfg


class CAMA(nn.Module):
    def __init__(self, dim_y, dim_z, dim_m):
        super(CAMA, self).__init__()

        # Set dimensions of hidden state representations
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.dim_m = dim_m

        self.encoder = VariationalEncoder(
            dim_y=self.dim_y, dim_z=self.dim_z, dim_m=self.dim_m,
        )
        self.decoder = Decoder(
            dim_y=self.dim_y, dim_z=self.dim_z, dim_m=self.dim_m,
        )

    def forward(self, x, y, infer_m=False):
        m, z = self.encoder(x, y, infer_m=infer_m)

        x_recon = self.decoder(y, z, m)

        return x_recon

    def predict(self, x):
        '''
        Approximates p(y|x) and returns probability distribution over
        classes as float64 tensor
        '''
        # Copy x to get all possible class assignments for each sample
        # -> [x_1, x_2, ..., x_128, x_1, x_2, ...]
        x_rep = x.repeat(self.dim_y, 1, 1, 1).to(cfg.DEVICE)
        y_rep = torch.diag(torch.ones(self.dim_y)).to(
            cfg.DEVICE)  # -> [y=1, y=2, ...]
        # -> [y=1, y=1, ..., y=2, y=2, ...]
        y_rep = y_rep.repeat_interleave(x.shape[0], dim=0)

        # Sample m
        m = self.encoder.qm(x_rep)

        # z ~ q(z_k | x, y_c, m)
        # TODO: implement more than one sample of z and take average
        z_k, log_q_z = self.encoder.qz(x_rep, y_rep, m, return_log_prob=True)

        # Estimate p(y|x)
        # Reconstruction probability p(x | y_c, z_k, m)
        x_rec = self.decoder(y_rep, z_k, m)
        # Treat x_rec as mean of p(x|) ~ N(x_rec, 1)
        p = torch.distributions.Normal(x_rec, torch.ones_like(x_rec))
        # Diagonal covariance matrix so joint is product of marginals
        log_p_x = p.log_prob(x_rep).type(torch.float64).sum(
            # Sum over [C, H, W]
            axis=(-1, -2, -3)
        ).reshape((-1, 1))

        # Uniform prior class probability p(y_c) = 1/num_classes
        log_p_y = (
            torch.ones((x_rep.shape[0], 1)).type(torch.float64) *
            (-torch.log(torch.tensor(self.dim_y).type(torch.float64)))
        ).to(cfg.DEVICE)

        # Prior probability p(z_k):
        # Diagonal covariance matrix so joint is product of marginals
        log_p_z = self.encoder.qz.std_normal.log_prob(
            z_k.type(torch.float64)
        ).sum(-1, keepdim=True)

        # Pre-softmax class posterior probability
        # -> [p(y=1|x_1), p(y=1|x_2), ...]
        pre_softmax_flat = log_p_x + log_p_y + log_p_z - log_q_z

        pre_softmax_transposed = pre_softmax_flat.reshape(
            (self.dim_y, x.shape[0])
        )
        pre_softmax = pre_softmax_transposed.transpose(0, 1)

        y_pred = F.softmax(pre_softmax, dim=-1)

        return y_pred
