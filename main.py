'''
Main entry point of program
'''
import os

from src.models.vision_cama.cama import CAMA
from src.scripts.train import train_model
from src.scripts.eval_performance import eval_model, eval_robust
from src.scripts.representations import compute_representations
from src.utils.argparser import parse_args
from src.utils.data import get_data
from src.utils import config as cfg
from src.utils.model import save_model, load_model


def main():
    '''
    Main function, which calls all other functions as needed
    '''
    args = parse_args()

    # Define model
    model = CAMA(
        dim_y=cfg.DIM_Y, dim_z=cfg.DIM_Z, dim_m=cfg.DIM_M
    ).to(cfg.DEVICE)

    train_loader, val_loader, test_loader, train_loader_pert, val_loader_pert,\
        test_loader_pert = get_data()

    if args.mode == 'train':
        model = train_model(model, train_loader, val_loader, train_loader_pert,
                            val_loader_pert)
        save_model(model, tag='final')

        print('Evaluating on clean test dataset:')
        eval_model(model, test_loader, verbose=True)
        print('Evaluating model on perturbed test data')
        eval_model(model, test_loader_pert, verbose=True)

    elif args.mode == 'finetune':
        model = load_model(model, args.trained_model)
        eval_model(model, test_loader)

    elif args.mode == 'test':
        model = load_model(model, args.trained_model)
        xe, nce, acc = eval_model(model, test_loader, verbose=True)
        xe_pert, nce_pert, acc_pert = eval_model(
            model, test_loader_pert, verbose=True
        )

        with open(os.path.join(cfg.OUT_DIR, 'test_results.txt'), 'w') as w:
            w.write(f'Test results on model {args.trained_model}\n')
            w.write(f'Clean test data - XE: {xe} NCE: {nce} Acc.: {acc}\n')
            w.write(f'Pert. test data - XE: {xe_pert} NCE: {nce_pert} Acc.:{acc_pert}\n')
    elif args.mode == 'eval':
        eval_robust(args.trained_model, model, test_loader)

    elif args.mode == 'repr':
        model = load_model(model, args.trained_model)
        compute_representations(
            model, train_loader, test_loader, args.trained_model
        )

    else:
        raise ValueError(f'Unrecognised program mode {args.mode}')


if __name__ == '__main__':
    main()
