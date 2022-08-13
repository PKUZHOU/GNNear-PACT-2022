#! /bin/bash
dataset_dir=./dataset/ # don't forget the slash
results_dir=./results/ # change this to a new dir, e.g., ./results_0820/

python simulation/GNNear_experiments.py --dataset_root_dir $dataset_dir --results_dir $results_dir --throughput
python simulation/GNNear_experiments.py --dataset_root_dir $dataset_dir --results_dir $results_dir --breakdown 
python simulation/GNNear_experiments.py --dataset_root_dir $dataset_dir --results_dir $results_dir --ieo 
python simulation/GNNear_experiments.py --dataset_root_dir $dataset_dir --results_dir $results_dir --shard 
python simulation/GNNear_experiments.py --dataset_root_dir $dataset_dir --results_dir $results_dir --window 
python simulation/GNNear_experiments.py --dataset_root_dir $dataset_dir --results_dir $results_dir --ratio 
python simulation/GNNear_experiments.py --dataset_root_dir $dataset_dir --results_dir $results_dir --rank 