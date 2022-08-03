# Artifact of 'GNNear' paper for PACT 2022

This is the artifact of paper "GNNear: Accelerating Full-Batch Training of Graph Neural Networks with Near-Memory Processing", which can reimplement the experiments in Sec.6.

## Requirements

- Python 3.7.10
- Packages
  ```
  pip install -U pip setuptools
  pip install -r requirements.txt
  ```

## Usage

You can run ```GNNear_experiments.py``` and set arguments to set which experiment you want to run:

  ```
  GNNear_experiments.py [--dataset_root_dir DATASET_ROOT_DIR]
                        [--throughput] [--breakdown] [--ieo] [--shard]
                        [--window] [--ratio] [--rank]
  ```
- ```--dataset_root_dir DATASET_ROOT_DIR```: Set the root directory to save datasets used in our experiments. The default directory is ```~/datasets/Graph/```. Our code will automatically download missing datasets.
  
- ```--throughput```: If this option is on, training throughput comparison experiment in Sec.6.2 will be conducted, and the results will be saved in ```./results/throughput/```.
  
- ```--breakdown```: If this option is on, speedup breakdown over CPU experiment in Sec.6.3 will be conducted, and the results will be saved in ```./results/breakdown/```.
  
- ```--ieo```: If this option is on, effect of IEO experiment in Sec.6.3 will be conducted, and the results will be saved in ```./results/ieo/```.
  
- ```--shard```: If this option is on, shard size exploration experiment in Sec.6.4 will be conducted, and the results will be saved in ```./results/shard/```.

- ```--window```: If this option is on, window size exploration experiment will be conducted, and the results will be saved in ```./results/window/```

- ```--ratio```: If this option is on, duplication ratio exploration experiment will be conducted, and the results will be saved in ```./results/ratio/```

- ```--rank```: If this option is on, ranks per DIMM experiment will be conducted, and the results will be saved in ```./results/rank/```

Note that these results is large (typically GB level) and we utilize mutiprocessing to parallize different cases, so we recommend to run one experiment every time to avoid OOM problem. 