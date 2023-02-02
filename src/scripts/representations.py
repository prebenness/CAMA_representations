import os

import torch
import torch.nn.functional as F
import numpy as np

import src.utils.config as cfg


def compute_representations(model, train_loader, test_loader, model_path):
    # Create output directory
    model_dir = os.path.split(model_path)[0]
    outpath = os.path.join(model_dir, 'representations')
    os.makedirs(outpath, exist_ok=True)

    split_dict = {'train': train_loader, 'test': test_loader}

    for split_name, loader in split_dict.items():
        print(f'Storing representations for {split_name} dataset')
        images, style, content = [], [], []

        for idx, (x, y) in enumerate(loader):

            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            y = F.one_hot(y, num_classes=cfg.DIM_Y).type(torch.float32)

            with torch.no_grad():
                m = model.encoder.qm(x)
                z = model.encoder.qz(x, y, m)

                h_y = model.decoder.py(y)
                h_z = model.decoder.pz(z)
                # TODO: Figure out whether we want to store m or h_m
                h_m = model.decoder.pm(m)

            images.append(x)
            style.append(h_z)
            content.append(h_y)

            if (idx + 1) % 10 == 0:
                print(f'Processed {idx + 1} of {len(loader)} batches')

        # Save images, contents and styles for test and train sets
        images = torch.cat(images, 0).detach().cpu().numpy()
        style = torch.cat(style, 0).detach().cpu().numpy()
        content = torch.cat(content, 0).detach().cpu().numpy()

        np.savez(os.path.join(outpath, f'content_{split_name}.npz'), content)
        np.savez(os.path.join(outpath, f'images_{split_name}.npz'), images)
        np.savez(os.path.join(outpath, f'style_{split_name}.npz'), style)

    print(f'Representations computed and stored under {outpath}')
