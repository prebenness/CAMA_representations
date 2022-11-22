# Toy VAE implementation in PyTorch

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt

from models import VariationalAutoEncoder


## Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(m, data, epochs=20):
  opt = torch.optim.Adam(m.parameters())

  for epoch in range(epochs):
    num_samples = 0
    tot_rec_loss, tot_kl_loss = 0, 0
    for x, _ in data:
      x = x.to(device)
      opt.zero_grad()

      x_rec = m(x)
      
      rec_loss = ( (x - x_rec) ** 2 ).sum()
      kl_loss = m.encoder.kl
      loss = rec_loss + kl_loss
      
      loss.backward()
      opt.step()

      tot_rec_loss += rec_loss.detach().clone()
      tot_kl_loss += kl_loss.detach().clone()
      num_samples += x.shape[0]

    print(f'Epoch {epoch + 1} - rec_loss: {tot_rec_loss / num_samples} - kl_loss: {tot_kl_loss / num_samples}')

  return m


def plot_latent(ae, data, num_batches=100):
  for i, (x, y) in enumerate(data):
    z = ae.encoder(x.to(device))
    z = z.to('cpu').detach().numpy()
    plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')

    if i > num_batches:
      plt.colorbar()
      plt.grid()
      plt.savefig(os.path.join('toy_vae', 'output', 'latent_space.png'))
      break


def main():
  # Get MNIST dataset
  data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
      os.path.join('data'),
      transform=torchvision.transforms.ToTensor(),
      download=True),
    batch_size=128,
    shuffle=True,
  )

  # Instantiate model
  data_shape = [1, 28, 28]
  ae = VariationalAutoEncoder(latent_dim=2, in_dims=data_shape, hdim=512).to(device)
  summary(ae, tuple(data_shape))

  # Train model
  ae = train(ae, data, epochs=40)

  # Plot latent space
  plot_latent(ae, data, num_batches=100)


if __name__ == '__main__':
  main()