import os
import argparse

import tqdm
import numpy as np
from torch_geometric.datasets import Reddit, Yelp


DIMM_num = 4 * 4  # DIMM_per_channel * channel_num
vec_dim = 256
vec_bit = {
    256: 9,
    128: 8
}
robbin_num = {
    256: 32,
    128: 64
}


def parse_args():
    parser = argparse.ArgumentParser(description='Script for partitioning Pyg\'s graph data')
    parser.add_argument('--dataset_root_path', type=str, default='../../dataset/', help='root directory to save all graph datasets')
    parser.add_argument('--dataset_name', type=str, default='Reddit', help='PyG dataset name: Yelp or Reddit')

    args = parser.parse_args()
    return args


def get_dataset(dataset_name, dataset_root_path):
    assert dataset_name in ['Yelp', 'Reddit'], "wrong dataset name!"
    if dataset_name == 'Yelp':
        return Yelp(dataset_root_path+dataset_name+'/')
    if dataset_name == 'Reddit':
        return Reddit(dataset_root_path+dataset_name+'/')


def partition_adj_matix(args):
    dataset = get_dataset(args.dataset_name, args.dataset_root_path)
    #print(dataset.data)
    #print(dataset.data.edge_index)

    edges = dataset.data.edge_index.tolist()
    node_num = dataset.data.y.shape[0]
    edge_num = dataset.data.num_edges
    # create contracted adjacent vector dict
    print('create adjacent matrix')
    adjacent_dict = dict(zip([str(i) for i in range(node_num)], [[] for _ in range(node_num)]))
    for i in tqdm.trange(edge_num):
        adjacent_dict[str(edges[0][i])].append(edges[1][i])

    # split nodes into DIMMs, using round-robbin strategy
    # round-robbin number depends on vector dim
    print('split adjacent matrix')
    DIMM_sub_adj_matrix = [{} for _ in range(DIMM_num)]
    DIMM_idx = 0
    for i in tqdm.trange(node_num):
        DIMM_sub_adj_matrix[DIMM_idx][str(i)] = np.array(adjacent_dict[str(i)], dtype=np.uint)
        if (i+1) % robbin_num[vec_dim] == 0:
            DIMM_idx = (DIMM_idx+1) % DIMM_num

    # save sub-adjacent matrixes of each DIMM
    print('save partitioned sub-matrixes in npz format')
    if not os.path.exists('./' + args.dataset_name + '/'):
        os.mkdir('./' + args.dataset_name + '/')
    for i in tqdm.trange(len(DIMM_sub_adj_matrix)):
        np.savez('./' + args.dataset_name + '/DIMM' + str(i) + '.npz', **DIMM_sub_adj_matrix[i])


if __name__ == '__main__':
    args = parse_args()
    partition_adj_matix(args)
