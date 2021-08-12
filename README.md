# Data Science for Social Good + Changing Cities Research Lab


## Overview


## Setup

Make sure you have the relevant packages installed, including PyTorch, Seaborn, and TensorboardX.

You must have a `datasets` directory in the project root. The directory structure should be as such:

- `datasets/`
  - `boston/`
    - `imgs/`
    - `train.csv`
    - `val.csv`
  - `detroit/`
    - `...`
  - `la/`
    - `...`

The `train.csv` and `val.csv` files should each contain a field called `trueskill_category`, which can take values from 0-3, corresponding to ratings of 1-4, respectively. The code will take care of turning the buckets into labels (i.e. low blight/high blight, no blight/blight, amount blight).

Additionally, the CSV files should contain a field called `image_name` whose values are the names of the images in the respective `imgs/` folder. For instance, if a row in `datasets/boston/train.csv` has an `image_name` value of `1234.png`, then the code expects there to be an image: `datasets/boston/imgs/1234.png`.

Finally, create two new, empty directories: `exps` and `tensorboard`. The folders are used to store the experiment info and the Tensorboard logs, respectively, for the experiments you run.

## Quick Run

Once you have everything set up, type in this command to train the best model on the Detroit data so far:

```sh
python train.py --upsample --name det_best --label_type pa --l2 0 --gpu
```

You will find the experiment metadata in `exps/detroit_best`, model checkpoints in `exps/detroit_best/checkpoints`, and Tensorboard logs in `tensorboard/detroit_best`.

## Training

Train a model using the `train.py` script. There are a number of command line arguments you can adjust:

- `name` **(required)**: Name of the experiment, used to create a subdirectory in `exps` and `tensorboard`.
- `label_type`: Used to turn the `trueskill_category` into a label. Can be one of:
  - `lh`: For "low/high", i.e. 0 if `trueskill_category` == (0, 1) else 1
  - `pa`: For "presence/absence", i.e. 0 if `trueskill_category` == 0 else 1
  - `mc`: For "multiclass", i.e. 0 if `trueskill_category` == (0, 1); 1 if `trueskill_category` == 2; and 2 `trueskill_category` == 3
  - `regr`: For "regression", i.e. use the TrueSkill score ('score' column) directly as label for regression
- `batch_size`: Batch size for the data loaders
- `epochs` or `-e`: The number of epochs to run (default: 100)
- `lr`: Learning rate for the optimizer
- `l2`: L2 regularization weight
- `weighted`: Whether or not to use class weights (i.e. higher weighted loss for instances that belong to low-count classes)
- `upsample`: Whether or not to upsample data (i.e. repeat instances from low-count classes so that all classes "appear balanced" during training). If the label type is `regr`, then samples are upsampled by the inverse frequency of the binned TrueSkill score (with 10 bins)
- `add_bos`: Whether or not to add the Boston images
- `pretrain`: If this is set to true, then the script will train a segmentation model on the Cityscapes dataset--more on this below
- `pretrain_path`: If this is not None, then the script will load a pretrained segmentation model from this path and fine-tune its encoder as the classifier for the buildings task.
- `gpu`: Whether or not to use GPU
- `data_path`: Custom `datasets` directory location, if it is not underneath the project root. The value passed in replaces the project root, i.e. if the data is under /home/users/datasets, then '/home/users/' would be passed to this argument.
- `score_transform`: Applies a custom score transformation onto the regression labels to align across cities. Can be:
  - `piecewise_linear [base city name]`: Uses a piecewise linear transformation based on each city's class boundaries (for more details, see the 'Piecewise Linear Score Alignment Model Memo'). All cities' scores are aligned to the base city, i.e. `piecewise_linear detroit` would align all scores to the TrueSkill scores in Detroit.

## Pre-Training

Pre-training a model on the Cityscapes dataset is simple. Make sure to have the dataset loaded and unzipped in the `datasets/cityscapes` folder, with the folders of images (by city) in a subdirectory called `imgs` and the folders of masks (by city) in a subdirectory called `masks`. Then, simply run a command as such:

```
python train.py --name pretrain_lr4e-3_nol2 --pretrain --l2 0 --gpu --lr 4e-3
```

The pre-trained model can then be consumed by the train script again, this time to fine-tune on the buildings dataset. For example, running `python train.py` with `--pretrain_path pretrain_lr4e-3_nol2` will use the model trained by the above command.

## Evaluation
Train a model using the `eval.py` script. The script will automatically load the model with experiment name `name` and perform evaluation on a few images. The script takes only two arguments:

- `name` **(required)**: Name of the experiment, used to create a subdirectory in `exps` and `tensorboard`.
- `num_preds`: The number of predictions to perform from the validation set.

## Grad-CAMs

Use the `gradcam.py` script to generate and save Grad-CAMs from the classification models you have trained. The model with name `name` will be loaded and the entire validation dataset will be iterated through. Depending on whether or not the prediction is a TP, FP, TN, or FN, an image of the example with the overlayed CAM will be saved to `[outdir]/{tp, fp, tn, fn}`. Here is an example command to get the CAMs:

```sh
python gradcam.py --name example_exp
```
