'''
Scripts for evaluating trained models
'''

import os

import torchmetrics.functional as F_metrics
import torch
import torch.nn.functional as F
import torchattacks

import src.utils.config as cfg
from src.utils.model import load_model


def eval_model(model, data, verbose=True):
    '''
    Evaluate a given model on a given dataset
    '''

    # Turn off gradient computation
    with torch.no_grad():
        h_max = 0
        num_samples, xe_loss, num_correct = 0, 0, 0
        for x, y in data:
            x = x.to(cfg.DEVICE)
            y = y.to(cfg.DEVICE)
            y = F.one_hot(y, num_classes=cfg.DIM_Y).type(torch.float32)

            # Follow NIST convention and clamp values in range
            # 0.000000001 to 0.999999999
            y_pred = model.predict(x)                       # p(y|x)
            y_pred = y_pred.clamp(min=1e-9, max=1-1e-9)
            y_pred = torch.log(y_pred / (1 - y_pred))       # p(y|x) as logit

            y_av = torch.ones(y.shape) / y.shape[-1]            # p(y)
            y_av = torch.log(y_av / (1 - y_av)).to(cfg.DEVICE)  # p(y) as logit

            xe_loss += F.cross_entropy(y_pred, y, reduction='sum')
            h_max += F.cross_entropy(y_av, y, reduction='sum')
            num_samples += y.shape[0]

            # Accuracy: number of correctly labeled samples
            int_labels = y.argmax(dim=-1)
            acc = F_metrics.accuracy(
                preds=y_pred, target=int_labels, task='multiclass',
                num_classes=y.shape[-1]
            )
            num_correct += acc * y.shape[0]

            if cfg.DEBUG:
                print('DEBUG set to True breaking eval after one batch!')
                break

    # Cross entropy
    mean_xe = xe_loss / num_samples
    mean_h_max = h_max / num_samples
    nce = (mean_h_max - mean_xe) / mean_h_max

    # Accuracy
    mean_acc = num_correct / num_samples

    if verbose:
        print(
            f'Mean XE loss: {mean_xe:.3f} - NCE: {nce:.3f} - Acc: {mean_acc:.3f}'
        )

    return mean_xe, nce, mean_acc


def eval_robust(model_path, model, test_loader):
    '''
    Evaluate accuracy under PGD-40 attack
    '''

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model.predict(x)

    num_corr, num_corr_adv, num_tot = 0, 0, 0
    for (x, y) in test_loader:
        x = x.to(cfg.DEVICE)
        y = y.to(cfg.DEVICE)

        # Load model
        model = load_model(model, model_path)
        wrapped_model = Wrapper(model).to(cfg.DEVICE)

        # Get clean predictions
        y_pred = wrapped_model(x)
        num_corr += (y_pred.argmax(dim=1) == y).sum().item()
        num_tot += y.shape[0]

        # Craft adversarial samples
        attack = torchattacks.PGD(
            wrapped_model, eps=8/255, alpha=4/255, steps=40, random_start=True
        )
        x_adv = attack(x, y)
        y_pred_adv = wrapped_model(x_adv)

        num_corr_adv += (y_pred_adv.argmax(dim=1) == y).sum().item()

        print(f'Processed {num_tot:5d} of {len(test_loader.dataset):5d} samples, current tally: Clean acc: {num_corr / num_tot:4.3f} Adv acc: {num_corr_adv / num_tot:4.3f}')

    # Compute and store results
    clean_acc = num_corr / num_tot
    adv_acc = num_corr_adv / num_tot

    res_text = f'Test completed: Clean acc: {clean_acc} Adv acc: {adv_acc}'

    model_dir, model_file = os.path.split(model_path)
    log_file_name = f'{model_file.split(".")[0]}-robust_log.txt'
    with open(os.path.join(model_dir, log_file_name), 'w') as w:
        w.write(res_text)

    print(res_text)
