#! /bin/bash

python simulation/GNNear_experiments.py --dataset_root_dir ./dataset/ --throughput
python simulation/GNNear_experiments.py --dataset_root_dir ./dataset/ --breakdown 
python simulation/GNNear_experiments.py --dataset_root_dir ./dataset/ --ieo 
python simulation/GNNear_experiments.py --dataset_root_dir ./dataset/ --shard 
python simulation/GNNear_experiments.py --dataset_root_dir ./dataset/ --window 
python simulation/GNNear_experiments.py --dataset_root_dir ./dataset/ --ratio 
python simulation/GNNear_experiments.py --dataset_root_dir ./dataset/ --rank 