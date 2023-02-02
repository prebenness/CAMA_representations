'''
Model training scripts
'''

import os

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from src.scripts.eval_performance import eval_model
import src.utils.config as cfg
from src.utils.model import save_model


def store_recons(x, x_rec, tag='sample'):
    '''
    Store reconstruction of random image
    '''
    x = x[0].detach().cpu()
    x_rec = x_rec[0].detach().cpu()

    out_dir = os.path.join(cfg.OUT_DIR, 'recons')
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, f'{tag}_image.png')
    save_image([x, x_rec], out_file)


def train_model(model, train_loader, val_loader,
                train_loader_pert, val_loader_pert):
    '''
    Train a DeepCAMA model using clean and perturbed data
    '''
    opt = torch.optim.Adam(model.parameters())

    for epoch in range(cfg.NUM_EPOCHS if not cfg.DEBUG else 1):

        # Track losses and metrics
        best_val_acc = float('-inf')
        num_samples_clean, num_samples_pert = 0, 0
        tot_rec_loss_clean, tot_kl_loss_clean, \
            tot_rec_loss_pert, tot_kl_loss_pert = 0, 0, 0, 0

        for (x_clean, y_clean), (x_pert, y_pert) in zip(train_loader, train_loader_pert):
            # Format data and send to GPU
            x_clean = x_clean.to(cfg.DEVICE)
            y_clean = y_clean.to(cfg.DEVICE)
            y_clean = F.one_hot(
                y_clean, num_classes=cfg.DIM_Y
            ).type(torch.float32)

            x_pert = x_pert.to(cfg.DEVICE)
            y_pert = y_pert.to(cfg.DEVICE)
            y_pert = F.one_hot(
                y_pert, num_classes=cfg.DIM_Y
            ).type(torch.float32)

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
        xe, nce, acc = eval_model(model, train_loader, verbose=False)
        val_xe, val_nce, val_acc = eval_model(model, val_loader, verbose=False)
        val_xe_pert, val_nce_pert, val_acc_pert = eval_model(
            model, val_loader_pert, verbose=False)

        # If best so far, store model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, tag='best')

        # Print performance to stdout
        train_result_text = ' '.join(f'''\
            TRAIN: Epoch {epoch + 1} \
            loss: {tot_loss / tot_samples:.3f} \
            rec_cl: {tot_rec_loss_clean / num_samples_clean:.3f} \
            kl_cl: {tot_kl_loss_clean / num_samples_clean:.3f} \
            rec_pt: {tot_rec_loss_pert / num_samples_pert:.3f} \
            kl_pt: {tot_kl_loss_pert / num_samples_pert:.3f} \
            XE: {xe:.3f} \
            NCE: {nce:.3f} \
            acc: {acc:.3f}
            '''.split())

        val_result_text = ' '.join(f'''
            VAL: Clean XE: {val_xe:.3f} \
            Clean NCE: {val_nce:.3f} \
            Clean Acc: {val_acc:.3f} \
            Pert. XE: {val_xe_pert:.3f} \
            Pert. NCE: {val_nce_pert:.3f} \
            Pert. acc: {val_acc_pert:.3f} 
            '''.split())

        print(train_result_text)
        print(val_result_text)

        # Store a random reconstructed image for qualitative performance
        store_recons(x_clean, x_rec_clean, tag=f'epoch={epoch+1}')

        # Store checkpointed models
        if (epoch + 1) % 5 == 0:
            save_model(model, tag=f'epoch={epoch+1}')

    return model
