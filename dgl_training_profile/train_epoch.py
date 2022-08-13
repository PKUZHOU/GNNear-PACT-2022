import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from torch.utils.data import dataset
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, GINConv,SAGEConv
from dgl.data import RedditDataset
from pynvml import *
import time
from ogb.nodeproppred import DglNodePropPredDataset
from graph_dataset import NELLDataset, YelpDataset
import argparse

class Net(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layer, config='gcn'):
        super(Net, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.config = config
        if config == 'gcn':
            for l in range(n_layer-1):
                if l == 0:
                    self.conv_layers.append(GraphConv(input_dim, hidden_dim))
                else:
                    self.conv_layers.append(GraphConv(hidden_dim, hidden_dim))
            self.conv_layers.append(GraphConv(hidden_dim, output_dim))
        
        elif config == 'gat':
            for l in range(n_layer-1):
                if l == 0:
                    self.conv_layers.append(GATConv(input_dim, hidden_dim, 1))
                else:
                    self.conv_layers.append(GATConv(hidden_dim, hidden_dim, 1))
            self.conv_layers.append(GATConv(hidden_dim, output_dim, 1))
        
        elif config == 'gin':
            for l in range(n_layer-1):
                if l == 0:
                    self.conv_layers.append(GINConv(
                        nn.Sequential(nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(True),
                        nn.Sequential(nn.Linear(hidden_dim, hidden_dim))), aggregator_type='sum'))
                else:
                    self.conv_layers.append(GINConv(
                        nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(True),
                        nn.Sequential(nn.Linear(hidden_dim, hidden_dim))), aggregator_type='sum'))
            self.conv_layers.append(GINConv(
                        nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(True),
                        nn.Sequential(nn.Linear(hidden_dim, output_dim))), aggregator_type='sum'))

        elif config == 'sage':
            for l in range(n_layer-1):
                if l == 0:
                    self.conv_layers.append(SAGEConv(input_dim, hidden_dim,'mean'))
                else:
                    self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim,'mean'))
            self.conv_layers.append(SAGEConv(hidden_dim, output_dim,'mean'))
         
        else:
            assert False, "unsupproted model config, %s" % config
        # self.conv1 = GCNConv(dataset.num_node_features, dataset.num_classes)

    def forward(self, g, features):
        h = features
        for conv_layer in self.conv_layers:
            h = conv_layer(g, h)
            h = F.relu(h)
        if self.config == 'gat':
            h = torch.sum(h,dim=1,keepdim=False)
        return F.log_softmax(h, dim=1)

def main(args):
    use_device = args.device
    dataset_name = args.dataset_name
    model_name = args.model_name
    n_layers = args.layers
    hidden = args.hidden_dim

    print("Use Device: {}".format(use_device))
    print("Dataset: {}".format(dataset_name))
    print("Model: {}".format(model_name))
    print("Layers: {}".format(n_layers))
    print("Hidden Dim: {}".format(hidden))

    if dataset_name == 'nell':
        num_node_features = 61278
        dataset = NELLDataset(raw_dir="./dataset/NELL", force_reload=True)

    elif dataset_name == 'reddit':
        num_node_features = 602
        dataset = RedditDataset(raw_dir="./dataset/Reddit")

    elif dataset_name == 'am':
        num_node_features = 100
        dataset = DglNodePropPredDataset(name='ogbn-products', root = "./dataset")

    elif dataset_name == 'yp':
        num_node_features = 300
        dataset = YelpDataset(raw_dir='./dataset/Yelp')

    if use_device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if dataset_name == 'reddit':
        model = Net(602, dataset.num_classes, hidden, n_layers,  model_name).to(device)
    else:
        model = Net(num_node_features, dataset.num_classes, hidden, n_layers,  model_name).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    try:
        g = dataset.graph[0].to(device)
    except:
        g = dataset.graph.to(device)
    g = dgl.add_self_loop(g)

    features = g.ndata['feat']
    try:
        labels = g.ndata['label']
    except:
        labels = torch.zeros(features.size(0)).long().to(device)
    
    try:
        train_mask = g.ndata['train_mask']
    except:
        train_mask = torch.ones_like(labels)
    model.train()
    model.to(device)

    torch.cuda.synchronize()
    start = time.time()
    for epoch in range(5):
        out = model(g, features)
        loss = loss = F.nll_loss(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    print("Time: ", (time.time() - start)/5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPU/CPU training time profiler on DGL')
    parser.add_argument('--dataset_root_dir', type=str, default='./dataset', help='root directory to save all graph datasets')
    parser.add_argument('--device', type=str, default="cpu",choices=['cpu','gpu'], help = "Training on CPU or GPU" )
    parser.add_argument('--layers', type=int, default=2, help='number of GCN layers')
    parser.add_argument('--dataset_name', type=str, default='nell', choices=['nell','reddit','am','yp'], help='The type of input dataset')
    parser.add_argument('--model_name', type=str, choices=['gcn','gin','gat','sage'], default='gcn', help='type of gnn algorithms')
    parser.add_argument('--hidden_dim', type=int, default=256, help="Dim of the hidden feature")
    args = parser.parse_args()

    main(args)