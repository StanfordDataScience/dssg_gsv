"""gradcam.py
-------------------------------------------------------------------------
Created by: Shubhang Desai
Date created: November 2020
Last revised: June 15, 2021
Project: GSV
Subproject: ml-buildings
-------------------------------------------------------------------------

Gets Grad-CAMs of images given a trained model.

Arguments
---------
name : str (required)
    Name of the experiment you want to get CAMs from--this should be the same as
    the `name` arg passed in for that experiment's train script.

gpu : bool
    Flag added if GPU training desired

outdir : str
    Name of the output folder

feature_module : str
target_layer : str

Example
-------
python gradcam.py --name example_exp

Inputs
------
Requires a datasets/ directory with images in an imgs/ subdirectory and val.csv

Outputs
-------
Creates the four [outdir]/{tp, tn, fp, fn} directories where the CAMs will be saved
"""

from data import *

import torch
import torch.nn as nn
from torchvision import models

import argparse, copy, cv2
import os, json
import numpy as np

def get_args():
    """Gets command-line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--feature_module', type=str, default='layer4')
    parser.add_argument('--target_layer', type=str, default='2')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--outdir', type=str, default='cams')

    return vars(parser.parse_args())


def prepare_dirs(args):
    """Prepares the necessary output directories.
    """

    os.makedirs(args['outdir'])
    os.makedirs(args['outdir'] + '/tp')
    os.makedirs(args['outdir'] + '/tn')
    os.makedirs(args['outdir'] + '/fp')
    os.makedirs(args['outdir'] + '/fn')


def get_gradcam(model, img, feature_module, target_layer):
    """Gets the CAM of an image given the module and target layer in question.
    """

    features, gradients = [], []
    layer = model._modules[feature_module]._modules[target_layer]
    layer.register_forward_hook(lambda module, inp, out: features.append(out.cpu().data.numpy()[0]))
    layer.register_backward_hook(lambda module, gradin, gradout: gradients.append(gradout[0].cpu().data.numpy()[0]))

    model.zero_grad()
    y_hat = model(img)
    pred = np.argmax(y_hat.cpu().data.numpy())
    pred_prob = y_hat[0, pred]
    pred_prob.backward(retain_graph=True)

    weights = np.mean(gradients[0], axis=(1, 2))
    cam = np.sum([w * f for w, f in zip(weights, features[0])], axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img.shape[2:]))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam, pred


def show_cam_on_image(img, cam):
    """Given an image and its CAM, returns CAM overlayed onto image.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_img = cv2.addWeighted(heatmap, 0.4, np.float32(img), 0.6, 0)

    return np.uint8(255 * cam_img)


def deprocess_img(img):
    """ Unnormalizes image
    See https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65
    """

    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


def create_gradcam_images(model, loader, args):
    print(len(loader))
    for i, (img, label) in enumerate(loader):
        print(i)
        if args['gpu']: img = img.cuda()

        cam, pred = get_gradcam(model, img, args['feature_module'], args['target_layer'])
        img = deprocess_img(img.squeeze(0).permute((1, 2, 0)).data.cpu().numpy())
        cam_img = show_cam_on_image(np.float32(img) / 255., cam)

        folder_name = ('t' if pred == label.item() else 'f') + ('p' if pred == 1 else 'n')
        cv2.imwrite(args['outdir'] + '/' + folder_name + '/' + str(i) + '_cam.jpg', cam_img)
        cv2.imwrite(args['outdir'] + '/' + folder_name + '/' + str(i) + '.jpg', img)

        del cam_img, img, label


def main():
    """Main driver function
    """

    args = get_args()
    prepare_dirs(args)
    exp_path = 'exps/' + args['name']

    args['batch_size'] = 1
    args['label_type'] = json.load(open(exp_path + '/args.csv', 'r'))['label_type']

    model_path = exp_path + '/state_dicts/'
    model_path += sorted([('0' if len(d.split('_')[0]) == 1 else '') + d for d in os.listdir(model_path)])[-1]

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3 if args['label_type'] == 'mc' else 2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    if args['gpu']: model = model.cuda()

    create_gradcam_images(model, get_loader('val', args), args)


if __name__ == "__main__":
    main()
