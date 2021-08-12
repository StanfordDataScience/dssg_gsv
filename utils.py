"""utils.py
-------------------------------------------------------------------------
Created by: Shubhang Desai
Date created: April 2021
Last revised: June 15, 2021
Project: GSV
Subproject: ml-buildings
-------------------------------------------------------------------------

Various utility functions used by other files in the repo.
"""


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import metrics
from tqdm.autonotebook import tqdm


def get_cm_image(cm):
    """
    Returns a confusion matrix image given a NumPY array of the matrix.
    """

    fig = sns.heatmap(pd.DataFrame(cm, index=range(len(cm)), columns=range(len(cm))), annot=True).get_figure()
    fig.canvas.draw()
    cm_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    cm_img = cm_img.reshape(fig.canvas.get_width_height()[::-1] + (3,)).transpose((2, 0, 1))
    plt.close()

    return cm_img


def run(mode, model, loader, optimizer, args, e, writer=None, num_preds=-1):
    """
    Evalutes the model

    Parameters
    ----------
    model : torch.nn.Module
        model to run

    loader : torch.utils.data.DataLoader
        loader for data

    optimizer : torch.optim optimizer
        optimizer for model parameters

    args : dict
        dictionary of arguments from get_args()

    e : int
        current epoch number

    writer : tensorboardX.SummaryWriter
        TensorBoard logging object

    num_preds : int
        number of predictions to perform if not in train mode

    Returns
    -------
    acc : float
        accuracy on the validation set
    """

    if mode == 'train': model.train()
    else: model.eval()

    weights = None
    if args['weighted'] and not args['pretrain']:
        weights = np.unique(loader.dataset.labels, return_counts=True)[1].astype(float)
        weights = torch.Tensor(weights.sum() / weights / len(weights))

    total_loss, imgs, y_true, y_pred = 0.0, [], [], []
    for i, (img, label) in enumerate(tqdm(loader, desc=f'{mode} {e+1}')):
        if i == num_preds: break

        if args['gpu']: img, label, weights = img.cuda(), label.cuda(), (weights.cuda() if weights is not None else None)
        if mode == 'train': optimizer.zero_grad()

        y_hat = model(img)
        if args['label_type'] == 'regr':
            #print(y_hat, torch.unsqueeze(label, 1))
            loss = F.mse_loss(y_hat, torch.unsqueeze(label, 1))
            pred = torch.squeeze(y_hat)
        else:
            loss = F.cross_entropy(y_hat, label, weight=weights)
            pred = torch.argmax(y_hat, dim=1)
        
        if mode == 'train':
            loss.backward()
            optimizer.step()

        y_pred.extend(pred.data.cpu().numpy())
        # To save memory, only save these data if not in time series mode
        if not args['time_series']:
            total_loss += loss.item() * img.shape[0]
            imgs.extend(img.data.cpu().numpy())
            y_true.extend(label.data.cpu().numpy())

    if num_preds != -1:
        return imgs, y_pred, y_true

    y_true, y_pred = np.array(y_true).reshape(-1), np.array (y_pred).reshape(-1)

    
    print('%s %i' % (mode, e+1))
    
    total_loss /= len(loader.dataset)
    print('Loss\t%0.3f' % total_loss)
    if writer:
        writer.add_scalar(mode + '/loss', total_loss, e)
    
    if args['label_type'] != 'regr':
        cm = metrics.confusion_matrix(y_true, y_pred)
        acc = np.diagonal(cm).sum().astype(float) / cm.sum().astype(float)
    
        if writer:
            writer.add_scalar(mode + '/acc', acc, e)
            writer.add_image(mode + '/cm', get_cm_image(cm), e)

        if not args['pretrain']:
            stats = metrics.classification_report(y_true, y_pred, output_dict=True)
            print(metrics.classification_report(y_true, y_pred))
            p, r, f1 = stats['weighted avg']['precision'], stats['weighted avg']['recall'], stats['weighted avg']['f1-score']
            if writer:
                writer.add_scalar(mode + '/precision', p, e)
                writer.add_scalar(mode + '/recall', r, e)
                writer.add_scalar(mode + '/f1', f1, e)
    
        print('Acc\t%0.3f' % acc)
        print('CM:')
        print(cm)

        return stats
        
    else:
        return total_loss
