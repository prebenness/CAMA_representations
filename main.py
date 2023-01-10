'''
Main entry point of program
'''
from src.models.single_modality.cama import CAMA
from src.scripts.train import train_model
from src.scripts.eval_performance import eval_model
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
    model_class = CAMA(dim_y=cfg.DIM_Y, dim_z=cfg.DIM_Z,
                       dim_m=cfg.DIM_M, out_shape=cfg.OUT_SHAPE).to(cfg.DEVICE)

    if args.mode == 'train':
        train_loader, test_loader, train_loader_pert, \
            test_loader_pert = get_data()
        model = train_model(model_class, train_loader, train_loader_pert)
        save_model(model, name='test')

    elif args.mode == 'finetune':
        model = load_model(model_class, args.trained_model)
        eval_model(model, test_loader)

    elif args.mode == 'test':
        ...

    elif args.mode == 'repr':
        ...

    else:
        raise ValueError(f'Unrecognised program mode {args.mode}')


if __name__ == '__main__':
    main()
