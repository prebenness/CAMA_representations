'''
Model training scripts
'''

import os

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from src.scripts.eval_performance import eval_model
import src.utils.config as cfg


def store_recons(x, x_rec, epoch):
    '''
    Store reconstruction of random image
    '''
    x = x[32].detach().cpu()
    x_rec = x_rec[32].detach().cpu()

    out_path = os.path.join('logs', f'epoch-{epoch}-img.png')
    save_image([x, x_rec], out_path)


def train_model(model, clean_data, pert_data):
    '''
    Train a DeepCAMA model using clean and perturbed data
    '''
    opt = torch.optim.Adam(model.parameters())

    for epoch in range(cfg.NUM_EPOCHS if not cfg.DEBUG else 1):

        # Track losses
        num_samples_clean, num_samples_pert = 0, 0
        tot_rec_loss_clean, tot_kl_loss_clean, \
            tot_rec_loss_pert, tot_kl_loss_pert = 0, 0, 0, 0

        for (x_clean, y_clean), (x_pert, y_pert) in zip(clean_data, pert_data):
            # Format data and send to GPU
            x_clean = x_clean.to(cfg.DEVICE)
            y_clean = y_clean.to(cfg.DEVICE)
            y_clean = F.one_hot(y_clean).type(torch.float32)

            x_pert = x_pert.to(cfg.DEVICE)
            y_pert = y_pert.to(cfg.DEVICE)
            y_pert = F.one_hot(y_pert).type(torch.float32)

            # Zero gradient
            opt.zero_grad()

            # Clean data
            x_rec_clean = model(x_clean, y_clean)
            rec_loss_clean = 0.5 * ((x_clean - x_rec_clean) ** 2).sum()
            kl_loss_clean = cfg.BETA * model.encoder.qz.kl

            # Perturbed data
            x_rec_pert = model(x_pert, y_pert, infer_m=True)
            rec_loss_pert = 0.5 * ((x_pert - x_rec_pert) ** 2).sum()
            kl_loss_pert = cfg.BETA * \
                (model.encoder.qz.kl + model.encoder.qm.kl)

            # Tally losses
            loss_clean = rec_loss_clean + kl_loss_clean
            loss_pert = rec_loss_pert + kl_loss_pert

            total_loss = cfg.LAMBDA * loss_clean + (1 - cfg.LAMBDA) * loss_pert

            total_loss.backward()
            opt.step()

            # Compute metrics
            tot_rec_loss_clean += rec_loss_clean.detach().clone()
            tot_kl_loss_clean += kl_loss_clean.detach().clone()
            num_samples_clean += x_clean.shape[0]

            tot_rec_loss_pert += rec_loss_pert.detach().clone()
            tot_kl_loss_pert += kl_loss_pert.detach().clone()
            num_samples_pert += x_pert.shape[0]

            if cfg.DEBUG:
                print('DEBUG set to True breaking training after one batch!')
                break

        # Average loss weighted by cfg.LAMBDA
        tot_loss = cfg.LAMBDA * (tot_rec_loss_clean + tot_kl_loss_clean) \
            + (1 - cfg.LAMBDA) * (tot_rec_loss_pert + tot_kl_loss_pert)
        tot_samples = (cfg.LAMBDA * num_samples_clean +
                       (1 - cfg.LAMBDA) * num_samples_pert)

        # Check predictive power of model

        xe, nce, acc = eval_model(model, clean_data, verbose=False)

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

        # DEBUG reconstruction
        # Display some random images
        store_recons(x_clean, x_rec_clean, epoch=epoch+1)

    return model
