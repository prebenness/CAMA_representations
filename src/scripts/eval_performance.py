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
    Call robustness evaluations and store results in logs
    '''

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model.predict(x)

    # Load and wrap model
    _model = load_model(model, model_path)
    model = Wrapper(_model).to(cfg.DEVICE)

    res_text = test_robustness(model, test_loader)

    model_dir, model_file = os.path.split(model_path)
    log_file_name = f'{model_file.split(".")[0]}-robust_log.txt'
    with open(os.path.join(model_dir, log_file_name), 'w') as w:
        w.write(res_text)


def test_robustness(model, data_loader):
    '''
    Test adversarial robustness under different attacks
    '''

    class DummyAttack():
        def __init__(self, m):
            ...

        def set_normalization_used(self, mean, std):
            ...

        def __call__(self, x, y):
            return x

    attack_factories = {
        'dummy_attacker': DummyAttack,
        'pgd20_linf': lambda m: torchattacks.PGD(m, eps=8/255, alpha=2/255, steps=20, random_start=True),
        'pgd40_linf': lambda m: torchattacks.PGD(m, eps=8/255, alpha=4/255, steps=40, random_start=True),
        'pgd20_l2': lambda m: torchattacks.PGDL2(m, eps=1.0, alpha=0.2, steps=20, random_start=True),
        'pgd40_l2': lambda m: torchattacks.PGDL2(m, eps=1.0, alpha=0.2, steps=40, random_start=True),
        'fgsm_linf': lambda m: torchattacks.FGSM(m, eps=8/255),
        'cw20_l2': lambda m: torchattacks.CW(m, c=1, kappa=0, steps=20),
        'cw40_l2': lambda m: torchattacks.CW(m, c=1, kappa=0, steps=40),
    }

    results = {}
    for attack_name, attack_factory in attack_factories.items():
        print(f'Testing on {attack_name}')
        clean_acc, adv_acc = test_attack(
            model, attack_factory=attack_factory, test_loader=data_loader
        )
        results[attack_name] = (clean_acc, adv_acc)

    res_text = '\n'.join(
        [f'{attack_name} clean acc: {v[0]:10.8f} adv acc: {v[1]:10.8f}' for attack_name,
            v in results.items()]
    )

    print(res_text)
    return res_text


def test_attack(model, attack_factory, test_loader):
    '''
    Test adversarial robustness for single attack
    '''
    num_corr, num_corr_adv, num_tot = 0, 0, 0
    for (x, y) in test_loader:
        x = x.to(cfg.DEVICE)
        y = y.to(cfg.DEVICE)

        # Get clean test preds
        y_pred = model(x)

        # Craft adversarial samples
        attacker = attack_factory(model)
        x_adv = attacker(x, y)
        y_pred_adv = model(x_adv)

        num_corr += (y_pred.argmax(dim=1) == y).sum().item()
        num_corr_adv += (y_pred_adv.argmax(dim=1) == y).sum().item()
        num_tot += y.shape[0]

        print(f'Processed {num_tot:5d} of {len(test_loader.dataset):5d} samples, current tally: Clean acc: {num_corr / num_tot:4.3f} Adv acc: {num_corr_adv / num_tot:4.3f}')

    # Compute and store results
    clean_acc = num_corr / num_tot
    adv_acc = num_corr_adv / num_tot

    res_text = f'Test completed: Clean acc: {clean_acc} Adv acc: {adv_acc}'
    print(res_text)

    return clean_acc, adv_acc
