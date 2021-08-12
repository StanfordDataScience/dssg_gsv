"""train.py
-------------------------------------------------------------------------
Created by: Shubhang Desai
Date created: October 2020
Last revised: June 15, 2021
Project: GSV
Subproject: ml-buildings
-------------------------------------------------------------------------

Main entry script to train a classification model.

Arguments
---------
name : str (required)
    Name of the experiment

gpu : bool
    Flag added if GPU training desired

num_classes : int
batch_size : int
epcohs : int
lr : float
l2 : float
weighted : bool
upsample : bool
add_bos : bool
pretrain : bool
pretrain_path : str

Example
-------
python train.py --name example_exp --gpu

Inputs
------
Requires a datasets/ directory with images in an imgs/ subdirectory and train.csv and val.csv and test.csv files

Outputs
-------
Creates tensorboard/[name] directory with logging events
Creates exps/[name] directory with hyperparamters
"""


from data import *
from models import *
from utils import *

import torch.optim as optim
from tensorboardX import SummaryWriter
import os, argparse, json
from tqdm.autonotebook import trange
import sys


def get_args():
    """Gets command-line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--label_type', type=str, default='mc')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--weighted', action='store_true')
    parser.add_argument('--upsample', action='store_true')
    parser.add_argument('--add_bos', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--cities', '-c', action="extend", nargs="+", type=str, default=None) # Overrides --add_bos
    parser.add_argument('--score_transform', action="extend", nargs="+", type=str, default=None) 

    return vars(parser.parse_args())


def prepare_dir(args):
    """Creates necessary experiment directories given arguments
    """

    if args['name'] == 'test': return

    os.makedirs('exps/' + args['name'])
    os.makedirs('exps/' + args['name'] + '/checkpoints')

    with open('exps/' + args['name'] + '/args.csv', 'w') as f:
        json.dump(args, f)


def train(model, train_loader, optimizer, args, e, writer=None):
    """
    Evalutes the model

    Parameters
    ----------
    model : torch.nn.Module
        model to evaluate

    train_loader : torch.utils.data.DataLoader
        loader for training data

    optimizer : torch.optim optimizer
        optimizer for model parameters

    args : dict
        dictionary of arguments from get_args()

    e : int
        current epoch number

    writer : tensorboardX.SummaryWriter
        TensorBoard logging object

    Returns
    -------
    acc : float
        accuracy on the validation set
    """

    run('train', model, train_loader, optimizer, args, e, writer=writer)

@torch.no_grad()
def eval(model, val_loader, args, e, writer=None):
    """
    Evalutes the model

    Parameters
    ----------
    model : torch.nn.Module
        model to evaluate

    val_loader : torch.utils.data.DataLoader
        loader for validation data

    args : dict
        dictionary of arguments from get_args()

    e : int
        current epoch number

    writer : tensorboardX.SummaryWriter
        TensorBoard logging object

    Returns
    -------
    acc : float
        accuracy on the validation set
    """

    return run('eval', model, val_loader, None, args, e, writer=writer)


def main():
    """Main driver function
    """

    args = get_args()
    prepare_dir(args)

    # get data loader
    train_loader = get_loader('train', args)
    val_loader = get_loader('val', args)

    # create model
    model = get_model(args)
    if args['gpu']: model = model.cuda()

    # get optim
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['l2'])

    # train model
    writer = SummaryWriter(logdir='tensorboard/' + args['name'])
    best_f1 = 0.0
    best_loss = sys.float_info.max
    for e in trange(args['epochs'], desc='Training', unit='epoch'):
        train(model, train_loader, optimizer, args, e, writer)
        metrics = eval(model, val_loader, args, e, writer)
        
        if args['label_type'] != 'regr':
            f1 = metrics['macro avg']['f1-score']
            if f1 >= best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), 'exps/'  + args['name'] + '/checkpoints/%i_%i.pth' % (e, best_f1*100))
        
        else:
            loss = metrics
            if loss <= best_loss:
                best_loss = loss
                torch.save(model.state_dict(), 'exps/'  + args['name'] + '/checkpoints/%i_%i.pth' % (e, best_loss*100))


if __name__=="__main__":
    main()
