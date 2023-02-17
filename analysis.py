'''
Hacky scripts for reading training logs
'''

import os, re
from matplotlib import pyplot as plt


def read_logs(model_dir, epochs=range(5, 155, 5)):
    adv_accs, clean_accs, inf_disents = [], [], []
    for i in epochs:
        model_name = f'epoch={i}_model'
        test_log = os.path.join(model_dir, f'{model_name}-test_log.txt')
        robust_log = os.path.join(model_dir, f'{model_name}-robust_log.txt')
        dc_log = os.path.join(model_dir, f'{model_name}-representations', 'DC_result.txt')
        iob_log = os.path.join(model_dir, f'{model_name}-representations', 'IoB_result.txt')

        with open(test_log, 'r') as tl:
            # Test results on model results/test_epochs=250_cifar10_2023-02-02_15-22-45/models/epoch=45_model.pt
            # Clean test data - XE: 4.553367652431252 NCE: -0.9775040561339761 Acc.: 0.3319000005722046
            # Pert. test data - XE: 4.241469516343628 NCE: -0.8420483063913484 Acc.:0.310699999332428
            # 

            clean_acc = float(tl.readlines()[1].split()[-1])
            clean_accs.append(clean_acc)

        with open(robust_log, 'r') as rl:
            # Test completed: Clean acc: 0.295 Adv acc: 0.1918

            adv_acc = float(rl.readline().split()[-1])
            adv_accs.append(adv_acc)

        inf_disent = []
        try:
            with open(dc_log, 'r') as dc:
                # 
                # Distance Correlation for /global/D1/homes/prebenmn/workspace/CAMA_representations/results/test_epochs=250_cifar10_2023-02-02_15-22-45/models/epoch=5_model-representations:
                # content and style: 0.10603761584695771
                # Image and content: 0.3626163795959816
                # Image and style: 0.8081499658762132
                # 
                
                lines = dc.readlines()
                dc_cs = float(lines[2].split()[-1])
                dc_xc = float(lines[3].split()[-1])
                dc_xs = float(lines[4].split()[-1])
                inf_disent = [dc_cs, dc_xc, dc_xs]
        except:
            pass

        try:
            with open(iob_log, 'r') as iob:
                # 
                # IoB metric for /global/D1/homes/prebenmn/workspace/CaaM_representations/2-StandardDatasets/results/2023-02-06_19-49-37_resnet18_ours_mnist/checkpoints/resnet18_ours_cbam_multi/resnet18_ours_mnist/representations-resnet18_ours_cbam_multi-116-best:
                # MSE Content Bias: tensor(0.8294, device='cuda:0')
                # MSE Content: tensor(0.4107, device='cuda:0')
                # MSE Style Bias: tensor(0.8339, device='cuda:0')
                # MSE Style: tensor(0.3926, device='cuda:0')
                # IoBc: tensor(2.0197, device='cuda:0')
                # IoBs: tensor(2.1243, device='cuda:0')

                lines = iob.readlines()
                iobc = float(re.sub('[^0-9\.]', '', lines[-2].split()[1]))
                iobs = float(re.sub('[^0-9\.]', '', lines[-1].split()[1]))
                inf_disent += [iobc, iobs]
        except:
            pass

        if inf_disent:
            inf_disents.append(compute_inf_disent(inf_disent))


    return clean_accs, adv_accs, inf_disents


def compute_inf_disent(metrics):
    '''
    metrics either [DC(C,S), DC(X,C), DC(X,S)] or
    [DC(C,S), DC(X,C), DC(X,S), IoB(X,C), IoB(X,S)]
    '''

    assert len(metrics) == 3 or len(metrics) == 5

    # DC variable rename
    metrics[0] = 1 - metrics[0]

    # IoB variable rename
    if len(metrics) == 5:
        metrics[3] = 1 - 1/metrics[3]
        metrics[4] = 1 - 1/metrics[4]

    return sum([ m**2 for m in metrics ]) ** 0.5 / len(metrics) ** 0.5


def plot_training(clean_accs, adv_accs, inf_disents, model_dir, epochs=range(5,155,5)):
    fontsize_large = 20
    fontsize_normal = 16
    inf_disent_epochs = [ [ *epochs ][i] for i, _ in enumerate(inf_disents) ]

    fig, ax = plt.subplots()
    ax.plot(epochs, clean_accs, color='royalblue', marker='.')
    ax.plot(epochs, adv_accs, color='orange', marker='.')
    ax.set_xlabel("Epoch", fontsize=fontsize_normal)
    ax.set_ylabel("Accuracy", fontsize=fontsize_normal)

    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(inf_disent_epochs, inf_disents, color='seagreen', linestyle='--', marker='.')
    ax2.set_ylabel("ID", fontsize=fontsize_normal)
    
    ax.grid()
    ax.set_xlim([5, 150])
    fig.legend(['Clean', 'PGD40', 'ID'], loc=[0.34, 0.27], fontsize=fontsize_normal)
    #ax.set_title('CAMA on CIFAR10', fontsize=fontsize_large)
    fig.subplots_adjust(bottom=0.13, left=0.14, right=0.85, top=0.95)
    fig.savefig(
        os.path.join(model_dir, 'CAMA_CIFAR10_train_plot.png'),
        dpi=300
    )


if __name__ == '__main__':
    model_dir = 'results/test_epochs=250_cifar10_2023-02-02_15-22-45/models'
    epochs = range(5, 155, 5)

    clean_accs, adv_accs, inf_disents = read_logs(
        model_dir=model_dir, epochs=epochs
    )
    plot_training(
        clean_accs=clean_accs, adv_accs=adv_accs, inf_disents=inf_disents,
        model_dir=model_dir, epochs=epochs,
    )




