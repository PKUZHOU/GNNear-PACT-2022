#! /bin/bash

dataset_root_dir=./dataset
dataset_name=reddit
model_name=gcn
layers=2
hidden_dim=256
device=cpu

python3 dgl_training_profile/train_epoch.py --dataset_root_dir $dataset_root_dir --dataset_name $dataset_name --model_name $model_name --device $device --layers $layers 