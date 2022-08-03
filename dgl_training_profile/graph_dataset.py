import imp
from dgl.data import DGLDataset
from torch_geometric.data import download_url
# from dataset import get_dataset
from torch_geometric.io import read_planetoid_data
from dgl.data.utils import generate_mask_tensor, load_graphs, save_graphs, deprecate_property
from dgl.convert import from_networkx
from torch_geometric.utils.convert import to_networkx
import os 
import torch.nn.functional as F
from torch_geometric.datasets import  Yelp
import shutil
import tarfile


class YelpDataset(DGLDataset):
    def __init__(self,raw_dir="./datasets/Graph/Yelp", force_reload=False, verbose=True):
        # self.num_node_features = 716847
        # self.graph = Yelp(raw_dir)
        super(YelpDataset, self).__init__(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose, name="Yelp",url=None)

    def process(self):
        # graph
        data = Yelp(self.raw_dir).data
        nx_data = to_networkx(data)
        self._graph = from_networkx(nx_data)
        # features and labels

        features = data.x
        labels = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        self._graph.ndata['train_mask'] = train_mask
        self._graph.ndata['val_mask'] = val_mask
        self._graph.ndata['test_mask'] = test_mask
        self._graph.ndata['feat'] = features
        self._graph.ndata['label'] = labels
        self._print_info()

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        if os.path.exists(graph_path):
            return True
        return False

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self._graph)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        graphs, _ = load_graphs(graph_path)
        self._graph = graphs[0]
        self._graph.ndata['train_mask'] = generate_mask_tensor(self._graph.ndata['train_mask'].numpy())
        self._graph.ndata['val_mask'] = generate_mask_tensor(self._graph.ndata['val_mask'].numpy())
        self._graph.ndata['test_mask'] = generate_mask_tensor(self._graph.ndata['test_mask'].numpy())
        self._print_info()

    def _print_info(self):
        if self.verbose:
            print('Finished data loading.')
            print('  NumNodes: {}'.format(self._graph.number_of_nodes()))
            print('  NumEdges: {}'.format(self._graph.number_of_edges()))
            print('  NumFeats: {}'.format(self._graph.ndata['feat'].shape[1]))
            print('  NumClasses: {}'.format(self.num_classes))

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 100

    @property
    def num_labels(self):
        deprecate_property('dataset.num_labels', 'dataset.num_classes')
        return self.num_classes

    @property
    def graph(self):
        deprecate_property('dataset.graph', 'dataset[0]')
        return self._graph

    @property
    def train_mask(self):
        deprecate_property('dataset.train_mask', 'graph.ndata[\'train_mask\']')
        return F.asnumpy(self._graph.ndata['train_mask'])

    @property
    def val_mask(self):
        deprecate_property('dataset.val_mask', 'graph.ndata[\'val_mask\']')
        return F.asnumpy(self._graph.ndata['val_mask'])

    @property
    def test_mask(self):
        deprecate_property('dataset.test_mask', 'graph.ndata[\'test_mask\']')
        return F.asnumpy(self._graph.ndata['test_mask'])

    @property
    def features(self):
        deprecate_property('dataset.features', 'graph.ndata[\'feat\']')
        return self._graph.ndata['feat']

    @property
    def labels(self):
        deprecate_property('dataset.labels', 'graph.ndata[\'label\']')
        return self._graph.ndata['label']


    def __getitem__(self, idx):
        r""" Get graph by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        :class:`dgl.DGLGraph`
            graph structure, node labels, node features and splitting masks:

            - ``ndata['label']``: node label
            - ``ndata['feat']``: node feature
            - ``ndata['train_mask']``： mask for training node set
            - ``ndata['val_mask']``: mask for validation node set
            - ``ndata['test_mask']:`` mask for test node set
        """
        assert idx == 0, "Reddit Dataset only has one graph"
        return self._graph

    def __len__(self):
        r"""Number of graphs in the dataset"""
        return 1

class NELLDataset(DGLDataset):
    def __init__(self,raw_dir=None, force_reload=False, verbose=False):
        url = 'http://www.cs.cmu.edu/~zhiliny/data/nell_data.tar.gz'
        super(NELLDataset, self).__init__(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose, name="NELL",url=url)
        # self.name = "NELL"
        self.num_node_features = 61278

    def process(self):
        # graph
        data = read_planetoid_data(os.path.join(self.raw_dir,'nell_data'), 'nell.0.001')
        nx_data = to_networkx(data)
        self._graph = from_networkx(nx_data)
        # features and labels

        features = data.x.to_dense()
        labels = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        self._graph.ndata['train_mask'] = train_mask
        self._graph.ndata['val_mask'] = val_mask
        self._graph.ndata['test_mask'] = test_mask
        self._graph.ndata['feat'] = features
        self._graph.ndata['label'] = labels
        self._print_info()

    def download(self):
        path = download_url(self.url, self.raw_dir)
        t = tarfile.open(path)
        t.extractall(path = self.raw_dir)
        # extract_tar(path, self.raw_dir)
        os.unlink(path)
        # shutil.rmtree(self.raw_dir)
        # os.rename(os.path.join(self.raw_dir, 'nell_data'), self.raw_dir)

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        if os.path.exists(graph_path):
            return True
        return False

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self._graph)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        graphs, _ = load_graphs(graph_path)
        self._graph = graphs[0]
        self._graph.ndata['train_mask'] = generate_mask_tensor(self._graph.ndata['train_mask'].numpy())
        self._graph.ndata['val_mask'] = generate_mask_tensor(self._graph.ndata['val_mask'].numpy())
        self._graph.ndata['test_mask'] = generate_mask_tensor(self._graph.ndata['test_mask'].numpy())
        self._print_info()

    def _print_info(self):
        if self.verbose:
            print('Finished data loading.')
            print('  NumNodes: {}'.format(self._graph.number_of_nodes()))
            print('  NumEdges: {}'.format(self._graph.number_of_edges()))
            print('  NumFeats: {}'.format(self._graph.ndata['feat'].shape[1]))
            print('  NumClasses: {}'.format(self.num_classes))
            print('  NumTrainingSamples: {}'.format(F.nonzero_1d(self._graph.ndata['train_mask']).shape[0]))
            print('  NumValidationSamples: {}'.format(F.nonzero_1d(self._graph.ndata['val_mask']).shape[0]))
            print('  NumTestSamples: {}'.format(F.nonzero_1d(self._graph.ndata['test_mask']).shape[0]))

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 210

    @property
    def num_labels(self):
        deprecate_property('dataset.num_labels', 'dataset.num_classes')
        return self.num_classes

    @property
    def graph(self):
        deprecate_property('dataset.graph', 'dataset[0]')
        return self._graph

    @property
    def train_mask(self):
        deprecate_property('dataset.train_mask', 'graph.ndata[\'train_mask\']')
        return F.asnumpy(self._graph.ndata['train_mask'])

    @property
    def val_mask(self):
        deprecate_property('dataset.val_mask', 'graph.ndata[\'val_mask\']')
        return F.asnumpy(self._graph.ndata['val_mask'])

    @property
    def test_mask(self):
        deprecate_property('dataset.test_mask', 'graph.ndata[\'test_mask\']')
        return F.asnumpy(self._graph.ndata['test_mask'])

    @property
    def features(self):
        deprecate_property('dataset.features', 'graph.ndata[\'feat\']')
        return self._graph.ndata['feat']

    @property
    def labels(self):
        deprecate_property('dataset.labels', 'graph.ndata[\'label\']')
        return self._graph.ndata['label']


    def __getitem__(self, idx):
        r""" Get graph by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        :class:`dgl.DGLGraph`
            graph structure, node labels, node features and splitting masks:

            - ``ndata['label']``: node label
            - ``ndata['feat']``: node feature
            - ``ndata['train_mask']``： mask for training node set
            - ``ndata['val_mask']``: mask for validation node set
            - ``ndata['test_mask']:`` mask for test node set
        """
        assert idx == 0, "Reddit Dataset only has one graph"
        return self._graph

    def __len__(self):
        r"""Number of graphs in the dataset"""
        return 1
