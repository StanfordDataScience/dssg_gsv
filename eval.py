"""eval.py
-------------------------------------------------------------------------
Created by: Shubhang Desai
Date created: October 2020
Last revised: June 25, 2021
Project: GSV
Subproject: ml-buildings
-------------------------------------------------------------------------

Entry script to evaluate a classification model.

Arguments
---------
name : str (required)
    Name of the experiment you want to evalutate--this should be the same as the
    `name` arg passed in for that experiment's train script.

num_preds: int
    Number of predictions to perform. The validation loader is not shuffled so
    this script deteriministically performs prediction on the first num_preds
    examples in the val set.

Example
-------
python eval.py --name example_exp --num_preds 12

Inputs
------
Requires a datasets/ directory with images in an imgs/ subdirectory and train.csv and val.csv and test.csv files
Requires also a folder named `name` in the exps/directory

Outputs
-------
Creates exps/[name]/preds directory with prediction outputs
"""

from data import *
from models import *
from utils import *

import os, argparse, json

from PIL import ImageFile


def get_args():
    """Gets command-line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--num_preds', type=int, default=10)
    parser.add_argument('--data_path', type=str) # no default here, so null string doesn't overwrite training argument data_path
    #parser.add_argument('--cities', '-c', action="extend", nargs="+", type=str, default=None)
    parser.add_argument('--city', '-c', type=str) # Takes only one city
    parser.add_argument('--time_series', '-t', action='store_true')
    
    curr_args = vars(parser.parse_args())

    # Get original args and override them with any new args that exist
    name = curr_args['name']
    args = json.load(open(os.path.join('exps', name, 'args.csv'), 'r'))
    args.update(curr_args)
    
    # Convert city argument to format expected by args.cities
    if args['city'] is not None:
        args['cities'] = [args['city']]

    # Set args.gpu to true
    args['gpu'] = True
     
    return args


@torch.no_grad()
def eval(model, val_loader, args):
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

    Returns
    -------
    acc : float
        accuracy on the validation set
    """

    return run('eval', model, val_loader, None, args, 0, None, num_preds=args['num_preds'])


def main():
    """Main driver function
    """
    # Fix for truncated images: https://stackoverflow.com/a/23575424
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    args = get_args()

    # get data loader
    val_loader = get_loader('time_series' if args['time_series'] else 'val', args)

    # create model
    model = get_model(args)
    if args['gpu']: model = model.cuda()

    model_path = 'exps/' + args['name'] + '/checkpoints/'
    model_path += sorted([('0' if len(d.split('_')[0]) == 1 else '') + d for d in os.listdir(model_path)])[-1]
    model.load_state_dict(torch.load(model_path))

    # eval model
    imgs, preds, gt = eval(model, val_loader, args)
    
    if args['time_series']:
        # Combine with original CSV
        csv = val_loader.dataset.csv
        csv['pred'] = preds
    
        # Write to preds.csv
        out_path = os.path.join(args['data_path'], 'datasets', args['city'], 'preds.csv')
        print(f'Writing to {out_path}...')
        csv.to_csv(out_path, index=False)
    else:
        np.save(os.path.join('exps', args['name'], 'imgs'), np.array(imgs))
        np.save(os.path.join('exps', args['name'], 'preds'), np.array(preds))
        np.save(os.path.join('exps', args['name'], 'gt'), np.array(gt))


if __name__=="__main__":
    main()
