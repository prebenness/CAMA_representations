import os

import torch
import torchvision
import torch.nn.functional as F

from models.single_modality.cama import CAMA


## Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
  # Set parameters of experiment
  dim_y, dim_z, dim_m = 10, 64, 32
  num_epochs = 10

  model = CAMA(dim_y=dim_y, dim_z=dim_z, dim_m=dim_m, out_shape=(1, 28, 28), device=device).to(device)

  data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
      os.path.join('data'),
      transform=torchvision.transforms.ToTensor(),
      download=True),
    batch_size=128,
    shuffle=True,
  )

  opt = torch.optim.Adam(model.parameters())

  for epoch in range(num_epochs):
    num_samples = 0
    tot_rec_loss, tot_kl_loss = 0, 0
    for x, y in data:
      x = x.to(device)
      y = F.one_hot(y, num_classes=dim_y).type(torch.float32)
      y = y.to(device)

      opt.zero_grad()

      x_rec = model(x, y)
      
      rec_loss = ( (x - x_rec) ** 2 ).sum()
      kl_loss = model.encoder.qz.kl
      loss = rec_loss + kl_loss
      
      loss.backward()
      opt.step()

      tot_rec_loss += rec_loss.detach().clone()
      tot_kl_loss += kl_loss.detach().clone()
      num_samples += x.shape[0]

    print(f'Epoch {epoch + 1} - rec_loss: {tot_rec_loss / num_samples} - kl_loss: {tot_kl_loss / num_samples}')


if __name__ == '__main__':
  main()