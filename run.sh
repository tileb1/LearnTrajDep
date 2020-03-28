#!/bin/bash

if [ -z "$SSH_AUTH_SOCK" ] ; then
    eval `ssh-agent`
    ssh-add
fi

git pull
source ../anaconda3/bin/activate
conda activate maowei

CUDA_VISIBLE_DEVICES=0 python main.py --epoch 5 --input_n 10 --output 10 --dct_n 20 --data_dir h3.6m/dataset/
 

git add .
git commit -m "auto commit from server after last run"
git push
