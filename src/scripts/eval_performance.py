'''
Scripts for evaluating trained models
'''

import torchmetrics.functional as F_metrics
import torch
import torch.nn.functional as F

import src.utils.config as cfg


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
            y = F.one_hot(y).type(torch.float32)

            # Follow NIST convention and clamp values in range 0.000000001 to 0.999999999
            y_pred = model.predict(x)                       # p(y|x)
            y_pred = y_pred.clamp(min=1e-9, max=1-1e-9)     # p(y|x) as logit
            y_pred = torch.log(y_pred / (1 - y_pred))

            y_av = torch.ones(y.shape) / y.shape[-1]        # p(y)
            y_av = torch.log(y_av / (1 - y_av)).to(cfg.DEVICE)  # p(y) as logit

            xe_loss += F.cross_entropy(y_pred, y, reduction='sum')
            h_max += F.cross_entropy(y_av, y, reduction='sum')
            num_samples += y.shape[0]

            # Accuracy: number of correctly labeled samples
            int_labels = y.argmax(dim=-1)
            acc = F_metrics.accuracy(preds=y_pred, target=int_labels,
                                     task='multiclass', num_classes=y.shape[-1])
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
            f'Mean XE loss: {mean_xe:.3f} - NCE: {nce:.3f} - Acc: {mean_acc:.3f}')

    return mean_xe, nce, mean_acc
