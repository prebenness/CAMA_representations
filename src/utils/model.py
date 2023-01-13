'''
Utility functions for storing and loading trained PyTorch models
'''
import os
from datetime import datetime

import torch

import src.utils.config as cfg


def save_model(model, name='model', results_dir=os.path.join('results')):
    '''
    Save model with name and unique timestamp
    '''

    name_list = [name, f'epochs={cfg.NUM_EPOCHS}', f'{cfg.DATASET}']

    if cfg.DEBUG:
        name_list = ['DEBUG'] + name_list

    name_list += [datetime.now().strftime('%Y-%m-%d_%H-%M-%S')]

    model_name = '_'.join(name_list)

    out_path = os.path.join(results_dir, model_name, 'trained')
    os.makedirs(out_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(out_path, 'model.pt'))


def load_model(model, path):
    '''
    Loads a stored model, must match given model class
    '''
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

    return model
