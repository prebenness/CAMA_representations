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
    tot_loss = 0
    for x, _ in data:
      x = x.to(device)
      opt.zero_grad()

      x_rec = m(x)
      
      loss = ( (x - x_rec) ** 2 ).mean() + m.encoder.kl
      loss.backward()
      opt.step()

      tot_loss += loss.detach().clone()


    av_loss = tot_loss / ( len(data) * data.batch_size )
    print(f'Epoch {epoch + 1} completed - average loss: {av_loss}')

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
  ae = train(ae, data, epochs=20)

  # Plot latent space
  plot_latent(ae, data, num_batches=100)


if __name__ == '__main__':
  main()