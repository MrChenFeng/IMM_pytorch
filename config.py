import yaml
import argparse
from pathlib import Path
import torch


def config(args=None):
    """
    Reads the command switches and creates a config
    Command line switches override config files
    :return:
    """

    """ config """
    parser = argparse.ArgumentParser(description='Parameters can be set by -c config.yaml or by positional params.')

    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('-d', '--device', type=str)
    parser.add_argument('--epochs', type=int)
    #parser.add_argument('--data_root', type=str, default='data')

    """ model parameters """
    parser.add_argument('--heatmap_std', type=int)
    parser.add_argument('--num_keypoints', type=int)

    """ hyper-parameters """
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)

    """ data and data augmentation parameters """
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--tps_control_pts', type=int)
    parser.add_argument('--tps_variance', type=float, help='0.01~0.1')
    parser.add_argument('--data_rescale_height', type=float)
    parser.add_argument('--data_rescale_width', type=float)

    args = parser.parse_args(args)

    def set_if_not_set(args, dict):
        for key, value in dict.items():
            if key in vars(args) and vars(args)[key] is None:
                vars(args)[key] = dict[key]
            elif key not in vars(args):
                vars(args)[key] = dict[key]
        return args

    if args.config is not None:
        with Path(args.config).open() as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            args = set_if_not_set(args, conf)

    defaults = {
        'num_keypoints': 10,
        'batch_size': 100,
        'dataset': 'AFLW',
        'tps_control_pts': 5,
        'lr': 0.01,
        'epochs': 20,
        'heatmap_std': 0.1,
        'tps_variance': 0.01,
        'data_rescale_height': 128,
        'data_rescale_width': 128,
        'split': 'random',
        'trainratio': 0.8,
        'testratio': 0.2
    }

    args = set_if_not_set(args, defaults)

    if args.device is None:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    return args


if __name__ == '__main__':
    t = config()
    from train import Trainer
    test = Trainer(t)

