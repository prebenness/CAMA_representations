'''
Utility functions for storing and loading trained PyTorch models
'''
import os

import torch

import src.utils.config as cfg


def save_model(model, tag='model'):
    '''
    Save model with name and unique timestamp
    '''

    out_path = os.path.join(cfg.OUT_DIR, 'models')
    os.makedirs(out_path, exist_ok=True)
    torch.save(
        model.state_dict(), os.path.join(
            out_path, f'{tag}_model.pt'
        )
    )


def load_model(model, path):
    '''
    Loads a stored model, must match given model class
    '''
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

    return model
