#!/bin/bash
#SBATCH --job-name=gsv-b_run
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH --gres gpu:1

#ml load opencv
#ml load py-pandas
#ml load py-scipystack
#ml load viz
#ml load py-pillow
#ml load py-scikit-learn/0.19.1_py27
#ml load torch
#pip install future
#pip install torchvision

ml load opencv
ml load py-pandas
ml load py-scipystack
ml load viz
ml load py-pillow
ml load torch
ml load py-pytorch/1.0.0_py27
ml load py-scikit-learn/0.19.1_py27

# Optional: Pretrain on Cityscapes
#python train.py --name pretrain_lr5e-3_nol2 --pretrain --l2 0 --gpu --lr 4e-3

# Train
python train.py --upsample --weighted --name det_multi_w_pretrained --label_type mc --l2 0 --pretrain_path exps/pretrain_lr1e-4_nol2 --gpu

# Eval
#python eval.py --name pretrain_default_nol2 --num_preds 2

# Get Grad-CAMs
#python gradcam.py --name det_lh_lr_1e-5_w
