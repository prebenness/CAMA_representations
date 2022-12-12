import os
from datetime import datetime

import torch
import torchvision
import torchmetrics.functional as F_metrics
import torch.nn.functional as F

from models.single_modality.cama import CAMA


## Use GPU if available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEBUG = False

def get_data(hor_shift=0, ver_shift=0):

  def transform(x):
    # Always normalise pixels to range [0.0, 1.0]
    x = torchvision.transforms.ToTensor()(x)
    x = torchvision.transforms.RandomAffine(degrees=0, translate=(hor_shift, ver_shift))(x)
    return x

  dataset = torchvision.datasets.MNIST(os.path.join('data'),
                                       transform=transform,
                                       download=True)

  data = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
  )

  return data


def train(model, clean_data, pert_data, num_epochs=10, lambd=0.7, beta=1.0):

  opt = torch.optim.Adam(model.parameters())

  for epoch in range(num_epochs if not DEBUG else 1):

    # Track losses
    num_samples_clean, num_samples_pert = 0, 0
    tot_rec_loss_clean, tot_kl_loss_clean, \
      tot_rec_loss_pert, tot_kl_loss_pert = 0, 0, 0, 0

    for (x_clean, y_clean), (x_pert, y_pert) in zip(clean_data, pert_data):
      ## Format data and send to GPU
      x_clean = x_clean.to(DEVICE)
      y_clean = y_clean.to(DEVICE)
      y_clean = F.one_hot(y_clean).type(torch.float32)

      x_pert = x_pert.to(DEVICE)
      y_pert = y_pert.to(DEVICE)
      y_pert = F.one_hot(y_pert).type(torch.float32)

      ## Zero gradient
      opt.zero_grad()

      # Clean data
      x_rec_clean = model(x_clean, y_clean)
      rec_loss_clean = 0.5 * ( (x_clean - x_rec_clean) ** 2 ).sum()
      kl_loss_clean = beta * model.encoder.qz.kl

      # Perturbed data
      x_rec_pert = model(x_pert, y_pert, infer_m=True)
      rec_loss_pert = 0.5 * ( (x_pert - x_rec_pert) ** 2).sum()
      kl_loss_pert = beta * (model.encoder.qz.kl + model.encoder.qm.kl)

      # Tally losses
      loss_clean = rec_loss_clean + kl_loss_clean
      loss_pert = rec_loss_pert + kl_loss_pert

      total_loss = lambd * loss_clean + (1 - lambd) * loss_pert

      total_loss.backward()
      opt.step()

      # Compute metrics
      tot_rec_loss_clean += rec_loss_clean.detach().clone()
      tot_kl_loss_clean += kl_loss_clean.detach().clone()
      num_samples_clean += x_clean.shape[0]

      tot_rec_loss_pert += rec_loss_pert.detach().clone()
      tot_kl_loss_pert += kl_loss_pert.detach().clone()
      num_samples_pert += x_pert.shape[0]

      if DEBUG:
        print('DEBUG set to True breaking training after one batch!')
        break

    # Average loss weighted by lambda
    tot_loss = lambd * (tot_rec_loss_clean + tot_kl_loss_clean) \
      + (1 - lambd) * (tot_rec_loss_pert + tot_kl_loss_pert)
    tot_samples = (lambd * num_samples_clean + (1 - lambd) * num_samples_pert)

    # Check predictive power of model

    xe, nce, acc = eval(model, clean_data, verbose=False)

    result_text = ' '.join(f'''\
      Epoch {epoch + 1} \
      loss: {tot_loss / tot_samples:.3f} \
      rec_cl: {tot_rec_loss_clean / num_samples_clean:.3f} \
      kl_cl: {tot_kl_loss_clean / num_samples_clean:.3f} \
      rec_pt: {tot_rec_loss_pert / num_samples_pert:.3f} \
      kl_pt: {tot_kl_loss_pert / num_samples_pert:.3f} \
      XE: {xe:.3f} \
      NCE: {nce:.3f} \
      acc: {acc:.3f}'''.split())

    print(result_text)


def eval(model, data, verbose=True):
  # Turn off gradient computation
  with torch.no_grad():
    h_max = 0
    num_samples, xe_loss, num_correct = 0, 0, 0
    for x, y in data:
      x = x.to(DEVICE)
      y = y.to(DEVICE)
      y = F.one_hot(y).type(torch.float32)

      # Follow NIST convention and clamp values in range 0.000000001 to 0.999999999
      y_pred = model.predict(x)                       # p(y|x)
      y_pred = y_pred.clamp(min=1e-9, max=1-1e-9)     # p(y|x) as logit
      y_pred = torch.log(y_pred / (1 - y_pred))

      y_av = torch.ones(y.shape) / y.shape[-1]        # p(y)
      y_av = torch.log(y_av / (1 - y_av)).to(DEVICE)  # p(y) as logit

      xe_loss += F.cross_entropy(y_pred, y, reduction='sum')
      h_max += F.cross_entropy(y_av, y, reduction='sum')
      num_samples += y.shape[0]

      # Accuracy: number of correctly labeled samples
      int_labels = y.argmax(dim=-1)
      acc = F_metrics.accuracy(preds=y_pred, target=int_labels, task='multiclass', num_classes=y.shape[-1])
      num_correct += acc * y.shape[0]

      if DEBUG:
        print('DEBUG set to True breaking eval after one batch!')
        break

  # Cross entropy
  mean_xe = xe_loss / num_samples
  mean_h_max = h_max / num_samples
  nce = (mean_h_max - mean_xe) / mean_h_max

  # Accuracy
  mean_acc = num_correct / num_samples

  if verbose:
    print(f'Mean XE loss: {mean_xe:.3f} - NCE: {nce:.3f} - Acc: {mean_acc:.3f}')

  return mean_xe, nce, mean_acc


def save(model, name='model'):
  if DEBUG:
    name_list = [ 'DEBUG', name ]
  else:
    name_list = [ name ]
  
  name_list += [ datetime.now().strftime('%Y-%m-%d_%H-%M-%S') ]
  
  model_name = '_'.join(name_list)

  out_path = os.path.join('results', model_name, 'trained')
  os.makedirs(out_path, exist_ok=True)

  torch.save(model.state_dict(), os.path.join(out_path, 'model.pt'))


def main():
  # Set parameters of experiment
  dim_y, dim_z, dim_m = 10, 64, 32
  num_epochs = 250

  clean_data = get_data()
  hor_shifted_data = get_data(hor_shift=0.01)
  ver_shifted_data = get_data(ver_shift=0.01)

  model = CAMA(dim_y=dim_y, dim_z=dim_z, dim_m=dim_m,
               out_shape=(1, 28, 28), device=DEVICE).to(DEVICE)

  # Train loop
  train(model, clean_data=clean_data, pert_data=hor_shifted_data,
        num_epochs=num_epochs, lambd=0.8, beta=10.0)

  # Eval model
  eval(model, clean_data)

  # Save model
  save(model, name='test')


if __name__ == '__main__':
  main()
