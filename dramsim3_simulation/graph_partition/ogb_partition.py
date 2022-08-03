import os
import argparse

import tqdm
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset


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
    parser = argparse.ArgumentParser(description='Script for partitioning OGB\'s graph data')
    parser.add_argument('--dataset_root_path', type=str, default='~/datasets/Graph/', help='root directory to save all graph datasets')
    parser.add_argument('--dataset_name', type=str, default='Amazon', help='PyG dataset name: Amazon or Protein')

    args = parser.parse_args()
    return args


def get_dataset(dataset_name, dataset_root_path):
    assert dataset_name in ['Amazon', 'Protein'], "wrong dataset name!"
    if dataset_name == 'Amazon':
        dataset = PygNodePropPredDataset(name="ogbn-products", root=dataset_root_path)
    if dataset_name == 'Protein':
        dataset = PygNodePropPredDataset(name="ogbn-proteins", root=dataset_root_path)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph = dataset[0]
    return graph


def partition_adj_matix(args):
    dataset = get_dataset(args.dataset_name, args.dataset_root_path)
    # print(dataset)
    # print(dataset.edge_index)

    edges = dataset.edge_index.tolist()
    node_num = dataset.y.shape[0]
    edge_num = dataset.num_edges
    redundant_edge_num = 0
    empty_node_num = 0

    # create contracted adjacent vector dict
    print('create adjacent matrix')
    adjacent_dict = dict(zip([str(i) for i in range(node_num)], [[] for _ in range(node_num)]))
    for i in tqdm.trange(edge_num):
        if len(adjacent_dict[str(edges[0][i])]) > 0 and adjacent_dict[str(edges[0][i])][-1] == edges[1][i]:
            redundant_edge_num += 1
            continue
        adjacent_dict[str(edges[0][i])].append(edges[1][i])

    # split nodes into DIMMs, using round-robbin strategy
    # round-robbin number depends on vector dim
    print('split adjacent matrix')
    DIMM_sub_adj_matrix = [{} for _ in range(DIMM_num)]
    DIMM_idx = 0
    for i in tqdm.trange(node_num):
        if len(adjacent_dict[str(i)]) == 0:
            empty_node_num += 1
            if empty_node_num == 1:
                print(i)
        DIMM_sub_adj_matrix[DIMM_idx][str(i)] = np.array(adjacent_dict[str(i)], dtype=np.uint)
        if (i+1) % robbin_num[vec_dim] == 0:
            DIMM_idx = (DIMM_idx+1) % DIMM_num

    # save sub-adjacent matrixes of each DIMM
    print("num nodes: %d, num edges: %d, num redundant edges: %d, num useful edges: %d, num empty nodes: %d" %
            (node_num, edge_num, redundant_edge_num, edge_num - redundant_edge_num, empty_node_num))
    print('save partitioned sub-matrixes in npz format')
    if not os.path.exists('./' + args.dataset_name + '/'):
        os.mkdir('./' + args.dataset_name + '/')
    for i in tqdm.trange(len(DIMM_sub_adj_matrix)):
        np.savez('./' + args.dataset_name + '/' + '/DIMM' + str(i) + '.npz', **DIMM_sub_adj_matrix[i])


if __name__ == '__main__':
    args = parse_args()
    partition_adj_matix(args)
