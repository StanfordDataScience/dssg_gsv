"""ImageDataset.py
-------------------------------------------------------------------------
Created by: Shubhang Desai
Date created: October 2020
Last revised: June 15, 2021
Project: GSV
Subproject: ml-buildings
-------------------------------------------------------------------------

Abstraction for a dataset of images to be used for training/eval.
"""


from torch.utils.data import Dataset
from torchvision import transforms
from .score_transforms import *

from PIL import Image
import pandas as pd
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, img_dirs, setname, label_type, args=None):
        """
        Initializes dataset of images

        Parameters
        ---------
        img_dirs : list
            directories which contains images in `imgs/` subdir and `[setname].csv` for labels

        setname : str
            one of ['train', 'val', 'test']

        label_type : str
            one of ['pa', 'lh', 'mc'] for presence/absence, low/high, and multi-class, respectively
        """

        label_fn = {
            'pa': lambda l: min(l, 1),
            'lh': lambda l: 0 if l <= 1 else 1,
            'mc': lambda l: max(l-1, 0),
            'regr': lambda l: l
        }[label_type]
        
        
        # getting baseline cutoffs to prepare for transformation
        if (args is not None) and 'score_transform' in args:
            score_transformation = {
                'piecewise_linear': lambda l, base_cuts, city_cuts: piecewise_linear(l, base_cuts, city_cuts)   
            }[args['score_transform'][0]]
            base_city = args['score_transform'][1]
            # always get the training set for the thresholds to transform to
            base_file = get_base_dir(img_dirs[0]) + base_city + '/train.csv' 
            base_cutoffs = get_cutoffs(pd.read_csv(base_file))
        
        self.images, self.labels = [], []
        for img_dir in img_dirs:
            self.csv = pd.read_csv(img_dir + setname + '.csv')

            self.images.extend([img_dir + 'imgs/' + i for i in self.csv['image_name'].values])
            if label_type == 'regr':
                new_scores = [np.float32(l) for l in self.csv['score'].values]
                if (args is not None) and 'score_transform' in args and get_city(img_dir) != base_city:
                    #city_cutoffs = get_cutoffs(self.csv) # change this line to load (pd.read_csv(img_dir + 'train.csv'))
                    city_cutoffs = get_cutoffs(pd.read_csv(img_dir+'train.csv')) # always get training set for thresholding
                    new_scores = [score_transformation(l, base_cutoffs, city_cutoffs) for l in new_scores]
                    new_scores = [np.float32(l) for l in new_scores]
                self.labels.extend(new_scores)
            else:
                self.labels.extend([label_fn(l) for l in self.csv['trueskill_category'].values])

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if setname == 'train':
            self.transform = transforms.Compose([
                #transforms.RandomResizedCrop(224),

                transforms.Resize(256),
                transforms.RandomCrop(224),
                
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=.05),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])


    def __len__(self):
        return len(self.images)


    def __getitem__(self, i):
        image = Image.open(self.images[i])
        image = self.transform(image)

        label = self.labels[i]

        return image, label
