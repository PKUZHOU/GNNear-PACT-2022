# Artifact of 'GNNear' paper for PACT 2022

This is the artifact of paper "GNNear: Accelerating Full-Batch Training of Graph Neural Networks with Near-Memory Processing", which can reimplement the experiments in Sec.6.


## Requirements

  You need to provide the following environment to run all of the Python scripts in our artifact.

- Python 3.7.10
- Packages
  ```
  pip install -U pip setuptools
  pip install -r requirements.txt
  ```

## Usage

### DGL Training Profile

We provide CPU/GPU training scripts using [DGL](https://www.dgl.ai/) framework in ```dgl_training_profile```. You can run ```train_epoch.py``` and set arguments to decide the configurations you want to run:

  ```
  python train_epoch.py [--dataset_root_dir DATASET_ROOT_DIR] [--device DEVICE] 
                        [--layers LAYERS] [--dataset_name DATASET_NAME]
                        [--model_name MODEL_NAME] [--hidden_dim HIDDEN_DIM]
  ```
- ```--dataset_root_dir DATASET_ROOT_DIR```: Set the root directory to save datasets used in our experiments. The default directory is ```./dataset```. Our code will automatically download missing datasets.

- ```--device DEVICE```: Choose the device (cpu or gpu) you want to run training task on. The default device is ```cpu```.

- ```--layers LAYERS```: Set the layer number of the model. The default layer number is 2.

- ```--dataset_name DATASET_NAME```: Choose the dataset you want to use. The default dataset is ```nell```.

- ```--model_name MODEL_NAME```: Choose the GNN model you want to train. The default model is ```GCN```.

-  ```--hidden_dim HIDDEN_DIM```: Set the input/output dim of hidden layers. The default dim is ```256```.

Alternatively, you can also modify and run  ```run_dhl_train_profile.sh``` to conduct these CPU/GPU training experiments.


### DRAMsim3 Simulation

To validate our architectue design, we implement a fine-grained simulator based on [DRAMsim3](https://github.com/umd-memsys/DRAMsim3) and [cnpy](https://github.com/rogersce/cnpy). This simulator can simulate one channel's DIMMs' behaviour on different graph datasets. 

To use this simulator, you should conduct the following steps:
- ```cd dramsim3_simulation/ext/cnpy```, then follow the ```README.md``` to generate dynamic library of cnpy. You can directly run ```compile.sh``` in this folder.

- ```cd graph_partition```, use the scripts to split graph data and save  them in ```.npz``` format. Currently, our simulator injects the meta-data of ```Amazon``` and ```Reddit``` datasets, you can also add other datasets in ```dramsim3_simulation/src/graph_settings.h```.

- Set the channel id and dataset you want to run by setting macros in ```dramsim3_simulation/src/graph_settings.h```.

- Compile the simulator using ```compile.sh``` in  ```dramsim3_simulation```, then run binary ```dramsim3main``` to start simulate.

You can see the results in ```dramsim3_simulation/results```. Each DIMM will report in DRAMsim3's manner.

Alternatively, you can also run ```run_dramsim3_simulation.sh``` to start simulation. The results will be saved in ```results``` in the main directory.

### Coarse-Grained Simulation

We provide Python scripts to fastly simulate DRAMSim3's behaviour in a corse-grained manner. You can run ```GNNear_experiments.py``` in ```simulation``` and set arguments to decide which experiment you want to run:

  ```
  python GNNear_experiments.py [--dataset_root_dir DATASET_ROOT_DIR]
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

Alternatively, you can also modify and run ```run_simulation.sh``` to directly conduct one of these experiments.

Note that these experiments will result in large memory occupation (typically GB level) and we utilize mutiprocessing to parallize different cases, so we recommend to run one experiment every time to avoid OOM problem. 