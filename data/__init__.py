"""__init__.py
-------------------------------------------------------------------------
Created by: Shubhang Desai
Date created: October 2020
Last revised: June 15, 2021
Project: GSV
Subproject: ml-buildings
-------------------------------------------------------------------------

Entry script for data package.
"""


from .ImageDataset import *
from .SegmentationDataset import *

from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from bisect import bisect_right

import os


def get_loader(setname, args):
    """
    Creates data loader to be used for training/eval

    Parameters
    ----------
    args : dict
        dictionary of arguments from get_args()

    Returns
    -------
    loader : torch.data.utils.DataLoader
        loader to be used for training/eval

    """
    
    def make_path(name):
        return os.path.join(args['data_path'], 'datasets', name) + '/'
    
    if args['pretrain']:
        dataset = SegmentationDataset(make_path('cityscapes'), setname)
    else:
        if 'cities' in args:
            # Validate args['cities']: cannot contain 'cityscapes', 'exps', 'tensorboard', or 'test'
            reserved_dirs = {'cityscapes', 'exps', 'tensorboard', 'test'}
            assert reserved_dirs.isdisjoint(args['cities']), f"Argument --cities cannot contain any of the following directories: {', '.join(reserved_dirs)}"
            datadir = [make_path(s) for s in args['cities']]
        else:
            datadir = [make_path('detroit')]
            if args['add_bos']:
                datadir += [make_path('boston')]  
        dataset = ImageDataset(datadir, setname, args['label_type'], args)      

    if setname == 'train' and args['upsample']:
        if args['label_type'] == 'regr':
            hist, bin_edges = np.histogram(dataset.labels)
            hist_weights = max(hist) / hist
            
            weights = [min(len(bin_edges)-1, bisect_right(bin_edges, i)) -1 for i in dataset.labels]
            weights = [hist_weights[i] for i in weights]

            sampler = WeightedRandomSampler(weights, len(weights))            
            
        else:
            weights = 1. / np.unique(dataset.labels, return_counts=True)[1].astype(float)
            weights = weights[dataset.labels]
            sampler = WeightedRandomSampler(weights, len(weights))
    else:
        sampler = None

    # Max out parallelism: 6 CPUs = 5 workers + 1 main thread
    return DataLoader(dataset, batch_size=args['batch_size'], shuffle=(setname=='train' and not args['upsample']), num_workers=5, sampler=sampler)
