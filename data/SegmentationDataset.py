"""SegmentationDataset.py
-------------------------------------------------------------------------
Created by: Shubhang Desai
Date created: April 2021
Last revised: June 15, 2021
Project: GSV
Subproject: ml-buildings
-------------------------------------------------------------------------

Abstraction for a Cityscape dataset to be used for pre-training.
"""


import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
import os, json

labels = ['rider', 'persongroup', 'motorcycle', 'traffic sign', 'road', 'car', 'trailer', 'wall', 'license plate', 'bicyclegroup', 'motorcyclegroup', 'ridergroup', 'pole', 'vegetation', 'ground', 'ego vehicle', 'out of roi', 'rectification border', 'sidewalk', 'train', 'person', 'polegroup', 'bridge', 'caravan', 'bus', 'dynamic', 'truckgroup', 'rail track', 'guard rail', 'sky', 'tunnel', 'bicycle', 'building', 'terrain', 'cargroup', 'truck', 'traffic light', 'fence', 'parking', 'static']

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, setname):
        """
        Initializes dataset of segmentation images/masks

        Paramters
        ---------
        img_dir : str
            directories which contains images in `imgs/` subdir and `[setname].csv` for labels

        setname : str
            one of ['train', 'val', 'test']
        """

        city_folders = [os.path.join(img_dir, 'imgs', setname, city) for city in os.listdir(os.path.join(img_dir, 'imgs', setname))]

        self.images = []
        for city_folder in city_folders: self.images.extend(os.path.join(city_folder, image_name) for image_name in os.listdir(city_folder))
        self.masks = [path.replace('imgs', 'masks').replace('leftImg8bit.png', 'gtFine_polygons.json') for path in self.images]

        self.transform = {
            'train': transforms.Compose([
                #transforms.RandomResizedCrop(224),

                transforms.Resize(256),
                transforms.CenterCrop(224),

                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }[setname]


    def __len__(self):
        return len(self.images)


    def __getitem__(self, i):
        image = Image.open(self.images[i])
        image = self.transform(image)

        mask_data = json.load(open(self.masks[i], 'r'))
        mask = Image.new('RGB', (mask_data['imgWidth'], mask_data['imgHeight']))
        draw = ImageDraw.Draw(mask)
        for obj in mask_data['objects']: draw.polygon([tuple(coord) for coord in obj['polygon']], fill=(labels.index(obj['label']), 0, 0))

        assert mask.size == (2048, 1024), 'Expected (2048, 1024), got ' + str(mask.size)
        mask = np.array(mask.resize((512, 256)))[16:16+224, 144:144+224, 0]

        return image, torch.Tensor(mask).long()
