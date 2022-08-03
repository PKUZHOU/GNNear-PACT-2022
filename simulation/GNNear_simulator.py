import math

import tqdm
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Reddit, Yelp

dataset_name = 'AM'

standard_frequency = 1e9


class Dataset:
    def __init__(self, dataset_name, dataset_root_path='~/datasets/Graph/') -> None:
        self.legal_datasets = ['RD', 'AM', 'PT', 'YP']
        self.dataset_name = dataset_name
        self.dataset_root_path = dataset_root_path
        self.graph = None
        self.input_feature_size = 0
        self.n_labels = 0
        self.get_dataset()

    def get_dataset(self):
        assert self.dataset_name in self.legal_datasets, "wrong dataset name!"

        if self.dataset_name == 'RD':
            self.input_feature_size = 602
            self.n_labels = 41
            self.graph = Reddit(self.dataset_root_path + 'Reddit' + '/')
        if self.dataset_name == 'YP':
            self.input_feature_size = 300
            self.n_labels = 100
            self.graph = Yelp(self.dataset_root_path + 'Yelp' + '/')
        if self.dataset_name == 'AM':
            self.input_feature_size = 100
            self.n_labels = 47
            self.graph = PygNodePropPredDataset(name="ogbn-products", root="~/datasets/Graph/")
        if self.dataset_name == 'PT':
            self.input_feature_size = 128
            self.n_labels = 112
            self.graph = PygNodePropPredDataset(name="ogbn-proteins")
            pass


class CAE:
    def __init__(self, is_GAT=False, GAT_attention_dim=16, is_GIN=False, GIN_mlp_layer=2, GIN_mlp_hidden_dim=256) -> None:
        self.array_size = 128
        self.n_vpu = 16
        self.vpu_lanes = 32
        self.frequency = 7e8

        self.is_GAT = is_GAT
        self.GAT_attention_dim = GAT_attention_dim
        self.is_GIN = is_GIN
        self.GIN_mlp_hidden_dim = GIN_mlp_hidden_dim
        self.GIN_mlp_layer = GIN_mlp_layer

    def calc_forward_cycles(self, in_feature_size, out_feature_size, n_vertex, last_layer):
        n_row_shards = math.ceil(in_feature_size*1.0 / self.array_size)
        n_col_shards = math.ceil(out_feature_size*1.0 / self.array_size)

        # for GIN, one network layer always have multiple MLP layers
        if self.is_GIN:
            n_hid_shards = math.ceil(self.GIN_mlp_hidden_dim*1.0 / self.array_size)
            total_rounds = n_vertex * n_hid_shards * (n_row_shards + (self.GIN_mlp_layer-2)*n_hid_shards + n_col_shards)
        # for GAT, we need to add cycles for calculating attention weight used in next layer
        elif self.is_GAT:
            if not last_layer:
                n_attention_shards = math.ceil(self.GAT_attention_dim*1.0 / self.array_size)
            else:
                n_attention_shards = 0
            total_rounds = n_vertex * n_col_shards * (n_row_shards + n_attention_shards)
        else:
            total_rounds = n_row_shards * n_col_shards * n_vertex

        return int(total_rounds * standard_frequency/self.frequency) 

    def calc_backward_cycles(self, in_feature_size, out_feature_size, n_vertex, last_layer):
        # Todo: more accurate estimation? ---- Now we emit the cost for updating weight and GAT's attention update
        n_row_shards = math.ceil(in_feature_size * 1.0 / self.array_size)
        n_col_shards = math.ceil(out_feature_size * 1.0 / self.array_size)

        if last_layer:
            # in last layer, we don't need to operate matmul, unless the model is GIN
            n_sub_vectors = math.ceil(in_feature_size*1.0 / self.vpu_lanes)
            n_process_rounds = math.ceil(n_sub_vectors*1.0 / self.n_vpu)
            total_rounds = n_process_rounds * n_sub_vectors * n_vertex
            if self.is_GIN:
                n_hid_shards = math.ceil(self.GIN_mlp_hidden_dim * 1.0 / self.array_size)
                total_rounds += n_vertex * n_hid_shards * ((self.GIN_mlp_layer-2)*n_hid_shards + n_col_shards)
        else:
            # the rest layer is similar to forward
            if self.is_GIN:
                n_hid_shards = math.ceil(self.GIN_mlp_hidden_dim * 1.0 / self.array_size)
                total_rounds = n_vertex * n_hid_shards * (n_row_shards + (self.GIN_mlp_layer-2)*n_hid_shards + n_col_shards)
            else:
                total_rounds = n_row_shards * n_col_shards * n_vertex
        return int(total_rounds * standard_frequency/self.frequency)

    def calc_update_weight_cycles(self, feature_size, n_vertex):
        n_sub_vectors = math.ceil(feature_size * 1.0 / self.vpu_lanes)
        n_process_rounds = math.ceil(n_sub_vectors * 1.0 / self.n_vpu)
        return int(n_vertex * n_process_rounds * standard_frequency/self.frequency)

    def calc_reduce_cycles(self, feature_size, n_vertex):
        n_sub_vectors = math.ceil(feature_size*1.0 / self.vpu_lanes)
        n_process_rounds = math.ceil(n_sub_vectors*1.0 / self.n_vpu)
        return int(n_vertex * n_process_rounds * standard_frequency/self.frequency)


class NME:
    def __init__(self, rank_num=2, window_size=1, dim_bytes=4, is_GIN=False, inter_schedule=True) -> None:
        self.rank_num = rank_num
        self.window_size = window_size
        self.frequency = 5e8

        # vector params
        self.feature_size = -1
        self.bytes = dim_bytes

        self.is_GIN = is_GIN
        self.inter_schedule = inter_schedule

        # shard property
        self.all_shards = {}
        self.total_shards = -1
        self.all_shard_cycles = []
        self.all_shard_first_load_cycles = []
        self.all_shard_last_compute_cycles = []
        self.all_shard_transfer_count = []
        self.all_load_count = 0
        self.all_transfer_count = 0
        self.all_compute_count = 0

        # NME PE params
        self.n_pe = 16
        self.pe_lanes = 8

    def calc_reduce_cycles(self, n_edges):
        assert self.feature_size > 0, "please set feature size first!"
        n_sub_vectors = math.ceil(self.feature_size*1.0 / self.pe_lanes)
        n_process_rounds = math.ceil(n_sub_vectors*1.0 / self.n_pe)
        if not self.is_GIN:
            n_process_rounds *= 2
        return int(n_edges * n_process_rounds * standard_frequency/self.frequency)

    def get_shard(self, shard_index):
        if shard_index not in self.all_shards.keys():
            return None
        return self.all_shards[shard_index]

    def simulate_full_reduce(self):
        self.all_shard_cycles.clear()
        self.all_shard_first_load_cycles.clear()
        self.all_shard_last_compute_cycles.clear()
        self.all_load_count = 0
        self.all_compute_count = 0

        for shard_index in tqdm.trange(self.total_shards):
            shard = self.get_shard(shard_index)
            if shard is not None:
                self.all_load_count += len(shard.keys())
                # first get load and compute cycles
                all_node_load_cycle = []
                all_node_compute_cycle = []
                for vertex, n_edge in shard.items():
                    # /2 represents duplicate-rank optimization, and *2 represents DDR technique
                    all_node_load_cycle.append(math.ceil(self.feature_size / self.rank_num * self.bytes * 1.0 / (8*2)))
                    all_node_compute_cycle.append(self.calc_reduce_cycles(n_edge))
                    self.all_compute_count += n_edge
                    # assert all_node_load_cycle[-1] >= all_node_compute_cycle[-1]

                # pipeline compute and load
                nme_cycles = 0
                for node_index in range(1, len(all_node_load_cycle)):
                    # assert all_node_compute_cycle[node_index-1] <= all_node_load_cycle[node_index]
                    if self.inter_schedule:
                        nme_cycles += max(all_node_compute_cycle[node_index-1], all_node_load_cycle[node_index])
                    else:
                        nme_cycles += all_node_compute_cycle[node_index-1] + all_node_load_cycle[node_index]
                self.all_shard_cycles.append(nme_cycles)
                self.all_shard_first_load_cycles.append(all_node_load_cycle[0])
                self.all_shard_last_compute_cycles.append(all_node_compute_cycle[-1])
            else:
                self.all_shard_cycles.append(0)
                self.all_shard_first_load_cycles.append(0)
                self.all_shard_last_compute_cycles.append(0)


class HGPGCNear:
    def __init__(self, channel=4, DIMMs_per_channel=4, rank_num=2, window_size=4, interval=128, dup_ratio=0.35,
                    is_GAT=False, is_GIN=False, change_order=True, use_NDP=True, inter_schedule=True) -> None:
        # system params
        self.channel = channel                                           # channel number
        self.channel_bits = int(math.log2(self.channel)) 
        self.DIMMs_per_channel = DIMMs_per_channel                       # DIMM number in each channel
        self.rank_num = rank_num                                         # rank number in each DIMM
        self.window_size = window_size                                   # slide window size for inter-shard scheduling
        self.ch_shift_pos = {
            256: int(4+math.log(self.rank_num, 2)),
            128: int(5+math.log(self.rank_num, 2)),
        }
    
        # GNN params
        self.input_feature_size = -1
        self.n_labels = -1
        self.hidden_size = 256
        self.bytes = 4
        self.n_layers = 2

        self.is_GAT = is_GAT
        self.GAT_attention_dim = 16
        self.is_GIN = is_GIN
        self.GIN_mlp_hidden_dim = 256
        self.GIN_mlp_layer = 2

        # graph & shard property
        self.edge_num = -1                   # edge number in this graph
        self.node_num = -1                   # node number in this graph
        self.interval = interval             # shard size
        self.total_shards = -1               # shard number for this graph
        self.shard_per_dimm = -1             # used for HGP
        self.shards = []                
        self.graph = None
        
        self.channel_read_cnt = []           # record how many times node need to be loaded in one channel
        self.max_channel_node_cnt = -1       # max node count in every channel
        self.max_channel_read_cnt = -1
        
        # system optimization settings
        self.dup_ratio = dup_ratio           # ratio of duplicated super nodes in each channel 
        self.change_order = change_order     # whether change aggregate-combine order in the first layer
        self.inter_shedule = inter_schedule  # whether use inter-shard scheduling
        self.use_NDP = use_NDP
        if not self.use_NDP:
            self.dup_ratio = 0
            self.DIMMs_per_channel = 1
            self.interval = 1
            self.window_size = 1
            self.inter_shedule = False

        # NME engines and CAE engine
        self.all_nmes = [[NME(self.rank_num, self.window_size, self.bytes, self.is_GIN, self.inter_shedule)
                          for _ in range(self.DIMMs_per_channel)] for _ in range(self.channel)]
        self.cae = CAE(self.is_GAT, self.GAT_attention_dim, self.is_GIN, self.GIN_mlp_layer, self.GIN_mlp_hidden_dim)

        # computation tasks in each DIMM
        for _ in range(self.channel):
            self.shards.append(dict(zip([i for i in range(self.DIMMs_per_channel)],
                                        [{} for _ in range(self.DIMMs_per_channel)])))

    ##################################### 
    #                                   #
    #          DIMM assignation         #
    #                                   #
    #####################################

    def get_ch_DIMM(self, src_index):
        ch_index = (src_index >> self.ch_shift_pos[self.hidden_size]) % self.channel
        DIMM_index = (src_index >> (self.ch_shift_pos[self.hidden_size] + self.channel_bits)) % self.DIMMs_per_channel
        return ch_index, DIMM_index

    def get_shard_index(self, dst_index):
        # may have empty shards (intervals)
        ori_shard_index = dst_index // self.interval
        mapped_shard_index = (ori_shard_index//self.shard_per_dimm) + self.DIMMs_per_channel*(ori_shard_index%self.shard_per_dimm)
        return mapped_shard_index
    
    def get_super_vertexes(self, edges):
        ch_vertex_degrees = [{} for _ in range(self.channel)]
        for i in tqdm.trange(self.edge_num):
            src_vertex = edges[0][i]
            ch_index, DIMM_index = self.get_ch_DIMM(src_vertex)
            if src_vertex not in ch_vertex_degrees[ch_index].keys():
                ch_vertex_degrees[ch_index][src_vertex] = 1
            else:
                ch_vertex_degrees[ch_index][src_vertex] += 1
        
        def take_second(elem):
            return elem[1]

        super_node_mask = [False for _ in range(self.node_num)]
        ch_degree_tuples = [[] for _ in range(self.channel)]
        for ch in range(self.channel):
            for vertex in ch_vertex_degrees[ch].keys():
                degree = ch_vertex_degrees[ch][vertex]
                ch_degree_tuples[ch].append((vertex, degree))
            ch_degree_tuples[ch].sort(key=take_second, reverse=True)

            ch_super_vertex_cnt = int(len(ch_degree_tuples[ch])*self.dup_ratio)
            for v in range(ch_super_vertex_cnt):
                vertex = ch_degree_tuples[ch][v][0]
                super_node_mask[vertex] = True

        return super_node_mask

    def assign_shards(self, dataset):
        self.graph = dataset.graph
        edges = self.graph.data.edge_index.tolist()

        self.edge_num = self.graph.data.num_edges
        self.node_num = self.graph.data.y.shape[0]
        self.total_shards = math.ceil(self.node_num * 1.0 / self.interval)
        self.shard_per_dimm = math.ceil(self.total_shards * 1.0 / self.DIMMs_per_channel)
        self.input_feature_size = dataset.input_feature_size
        self.n_labels = dataset.n_labels

        self.channel_read_cnt = [0 for _ in range(self.channel)]
        channel_merge_dict = [{} for _ in range(self.channel)]

        print("Assigning shards:")
        # step1: mark super nodes which need to be duplicated in each channel
        super_vertex_mask = self.get_super_vertexes(edges)

        # step2: assign shard for each DIMM --- for duplicated nodes, assign shard to corresponding DIMMs
        for i in tqdm.trange(self.edge_num):
            src_vertex = edges[0][i]
            dst_vertex = edges[1][i]
            ch_index, DIMM_index = self.get_ch_DIMM(src_vertex)
            shard_index = self.get_shard_index(dst_vertex)
            # for super nodes, calculation is assigned to spetial DIMMs
            if super_vertex_mask[src_vertex]:
                DIMM_index = shard_index % self.DIMMs_per_channel

            DIMM_shards_dict = self.shards[ch_index][DIMM_index]
            if shard_index in DIMM_shards_dict.keys():
                shard_dict = DIMM_shards_dict[shard_index]
                # src vertexes shoud be loaded in each shard
                if src_vertex in shard_dict.keys():
                    shard_dict[src_vertex] += 1
                else:
                    shard_dict[src_vertex] = 1
            else:
                DIMM_shards_dict[shard_index] = {src_vertex: 1}
            
            # record merge results
            if self.use_NDP:
                if dst_vertex not in channel_merge_dict[ch_index].keys():
                    channel_merge_dict[ch_index][dst_vertex] = [False for _ in range(self.DIMMs_per_channel)]
                channel_merge_dict[ch_index][dst_vertex][DIMM_index] = True
            else:
                self.channel_read_cnt[ch_index] += 1
        
        # step3: calculate merge read counts in each channel
        if self.use_NDP:
            for ch in range(self.channel):
                for merge_result in channel_merge_dict[ch].values():
                    self.channel_read_cnt[ch] += merge_result.count(True)
        self.max_channel_read_cnt = max(self.channel_read_cnt)
        self.max_channel_node_cnt = math.ceil(self.node_num/self.channel)

        # send assignation results to corresponding DIMMs
        for ch_index in range(self.channel):
            for DIMM_index in range(self.DIMMs_per_channel):
                self.all_nmes[ch_index][DIMM_index].total_shards = self.total_shards
                self.all_nmes[ch_index][DIMM_index].all_shards = self.shards[ch_index][DIMM_index]

    ##################################### 
    #                                   #
    #          cycle estimation         #
    #                                   #
    #####################################

    def calculate_NME_reduce_cycles(self, feature_size):
        # first calculate all DIMMs' processing cycles in all shards
        # when computing, apply inner-shard scheduling policy
        for ch_index in range(self.channel):
            for DIMM_index in range(self.DIMMs_per_channel):
                self.all_nmes[ch_index][DIMM_index].feature_size = feature_size
                self.all_nmes[ch_index][DIMM_index].simulate_full_reduce()

        total_cycles = 0
        sync_shard_index = -1
        finish_DIMM_count = 0
        is_DIMM_finished = [[False for _ in range(self.DIMMs_per_channel)] for _ in range(self.channel)]
        finish_shard_index = [[-1 for _ in range(self.DIMMs_per_channel)] for _ in range(self.channel)]
        cumulated_shard_finish_cycle = [[[] for _ in range(self.DIMMs_per_channel)] for _ in range(self.channel)]

        # then use sliding window strategy to schedule inter-shard latency
        while sync_shard_index < self.total_shards:
            if finish_DIMM_count == self.channel*self.DIMMs_per_channel:
                break

            # update end-point of every DIMM
            end_shard = min(self.total_shards, sync_shard_index+self.window_size + 1)
            for ch_index in range(self.channel):
                for DIMM_index in range(self.DIMMs_per_channel):
                    # in this DIMM has finished all shards, skip it
                    if finish_shard_index[ch_index][DIMM_index] == self.total_shards-1:
                        if not is_DIMM_finished[ch_index][DIMM_index]:
                            is_DIMM_finished[ch_index][DIMM_index] = True
                            finish_DIMM_count += 1
                        continue

                    # else, update as many shards as possible --- the boundary is constrained by sync_shard_id+window_size
                    for shard_index in range(finish_shard_index[ch_index][DIMM_index]+1, end_shard):
                        # get tmp_shard's process cycles
                        # last shard need to add last compute latency
                        if shard_index == self.total_shards-1:
                            shard_time = self.all_nmes[ch_index][DIMM_index].all_shard_first_load_cycles[shard_index] + \
                                         self.all_nmes[ch_index][DIMM_index].all_shard_cycles[shard_index] + \
                                         self.all_nmes[ch_index][DIMM_index].all_shard_last_compute_cycles[shard_index]
                        else:
                            shard_time = self.all_nmes[ch_index][DIMM_index].all_shard_first_load_cycles[shard_index] + \
                                         self.all_nmes[ch_index][DIMM_index].all_shard_cycles[shard_index]

                        # calculate cumulated finish cycles of tmp shard
                        if shard_index == finish_shard_index[ch_index][DIMM_index]+1:
                            if shard_index == 0:
                                start_cycle = total_cycles
                            else:
                                start_cycle = max(total_cycles, cumulated_shard_finish_cycle[ch_index][DIMM_index][-1])
                            cumulated_shard_finish_cycle[ch_index][DIMM_index].append(start_cycle+shard_time)
                        else:
                            cumulated_shard_finish_cycle[ch_index][DIMM_index].append(cumulated_shard_finish_cycle[ch_index][DIMM_index][-1]+shard_time)
                    finish_shard_index[ch_index][DIMM_index] = end_shard-1

            # find the fastest DIMM which firstly reaches the end-point and record the time
            min_shard_time = cumulated_shard_finish_cycle[0][0][end_shard-1]
            for ch_index in range(self.channel):
                for DIMM_index in range(self.DIMMs_per_channel):
                    min_shard_time = min(min_shard_time, cumulated_shard_finish_cycle[ch_index][DIMM_index][end_shard-1])

            # find the last shard_id which all DIMMs can finish before the earliest end-point time
            max_sync_shard = sync_shard_index+self.window_size
            for shard_index in range(sync_shard_index+1, end_shard):
                tmp_max_shard_time = cumulated_shard_finish_cycle[0][0][shard_index]
                for ch_index in range(self.channel):
                    for DIMM_index in range(self.DIMMs_per_channel):
                        tmp_max_shard_time = max(tmp_max_shard_time, cumulated_shard_finish_cycle[ch_index][DIMM_index][shard_index])
                if tmp_max_shard_time > min_shard_time:
                    max_sync_shard = shard_index-1
                    break

            # if we can sync at least one new shard, use min_shard_time to record total_cycle
            # else, some DIMMs need to wait and we force the sync go forward one step
            if max_sync_shard > sync_shard_index:
                sync_shard_index = max_sync_shard
                total_cycles = min_shard_time
            else:
                sync_shard_index += 1
                for ch_index in range(self.channel):
                    for DIMM_index in range(self.DIMMs_per_channel):
                        total_cycles = max(total_cycles, cumulated_shard_finish_cycle[ch_index][DIMM_index][sync_shard_index])

        # total cycled used in NME is the max cumulated cycle of the last sahrd
        for ch_index in range(self.channel):
            for DIMM_index in range(self.DIMMs_per_channel):
                total_cycles = max(total_cycles, cumulated_shard_finish_cycle[ch_index][DIMM_index][-1])
        return total_cycles

    def calculate_CAE_read_cycles(self, feature_size):
        # burst size is 8bytes/cycle, *2 means using DDR technique
        return math.ceil(feature_size*self.bytes*self.max_channel_read_cnt / (8*2))

    def calculate_CAE_merge_cycles(self, feature_size):
        # only contains calculation time consumption on CAE
        return self.cae.calc_reduce_cycles(feature_size, self.node_num)

    def calculate_CAE_forward_cycles(self, in_feature_size, out_feature_size, last_layer):
        return self.cae.calc_forward_cycles(in_feature_size, out_feature_size, self.node_num, last_layer)

    def calculate_CAE_backward_cycles(self, in_feature_size, out_feature_size, last_layer):
        return self.cae.calc_backward_cycles(in_feature_size, out_feature_size, self.node_num, last_layer)

    def calculate_CAE_weight_update_cycles(self, feature_size):
        # only contains calculation time consumption on CAE
        return self.cae.calc_update_weight_cycles(feature_size, self.node_num)

    def calculate_CAE_writeback_cycles(self, feature_size):
        return math.ceil(feature_size*self.bytes*self.max_channel_node_cnt / (8*2))

    def calculate_forward_cycles(self):
        self.forward_nme_cycles = []
        self.forward_cae_cycles = []

        for layer_index in range(self.n_layers):
            last_layer = False
            # aggregate
            if layer_index == 0:
                # first layer
                in_feature_size = self.input_feature_size
                out_feature_size = self.hidden_size
            elif layer_index == self.n_layers - 1:
                last_layer = True
                in_feature_size = self.hidden_size
                out_feature_size = self.n_labels
            else:
                in_feature_size = self.hidden_size
                out_feature_size = self.hidden_size
            print("layer %d's forward simulation starts" % layer_index)

            # change order for the first layer
            if self.change_order and self.input_feature_size > 256 and layer_index==0:
                print("Use change order optimization in the first layer")
                layer_forward_cycle = self.calculate_CAE_forward_cycles(in_feature_size, out_feature_size, last_layer)
                print("layer %d's cae forward cycle count is %d" % (layer_index, layer_forward_cycle))
                layer_merge_cycle = self.calculate_CAE_merge_cycles(out_feature_size)
                print("layer %d's cae merge cycle count is %d" % (layer_index, layer_merge_cycle))
                # read: original features, write: features multiplied by weight
                layer_read_write_cycle0 = self.calculate_CAE_writeback_cycles(in_feature_size) + self.calculate_CAE_writeback_cycles(out_feature_size)
                layer_read_write_cycle1 = self.calculate_CAE_writeback_cycles(out_feature_size) + self.calculate_CAE_writeback_cycles(out_feature_size)
                print("layer %d's 1st cae read & write cycle count is %d" % (layer_index, layer_read_write_cycle0))
                print("layer %d's 2nd cae read & write cycle count is %d" % (layer_index, layer_read_write_cycle1))
                layer_cae_cycle0 = max(layer_forward_cycle, layer_read_write_cycle0)
                layer_cae_cycle1 = max(layer_merge_cycle, layer_read_write_cycle1)

                layer_nme_cycle = self.calculate_NME_reduce_cycles(out_feature_size)
                print("layer %d's nme cycle count is %d" % (layer_index, layer_nme_cycle))

                self.forward_nme_cycles.append((0, layer_nme_cycle))
                self.forward_cae_cycles.append((layer_cae_cycle0, layer_cae_cycle1))
            else:
                layer_nme_cycle = self.calculate_NME_reduce_cycles(in_feature_size)
                print("layer %d's nme cycle count is %d" % (layer_index, layer_nme_cycle))

                layer_merge_cycle = self.calculate_CAE_merge_cycles(in_feature_size)
                print("layer %d's cae merge cycle count is %d" % (layer_index, layer_merge_cycle))
                layer_forward_cycle = self.calculate_CAE_forward_cycles(in_feature_size, out_feature_size, last_layer)
                print("layer %d's cae forward cycle count is %d" % (layer_index, layer_forward_cycle))
                layer_cae_compute_cycle = layer_merge_cycle + layer_forward_cycle

                # read partial sum
                layer_read_cycle = self.calculate_CAE_read_cycles(in_feature_size)
                print("layer %d's read cycle count is %d" % (layer_index, layer_read_cycle))
                # a_v^l, z_v^l and h_v^{l+1} need to be written back
                layer_write_back_cycle = self.calculate_CAE_writeback_cycles(in_feature_size) + self.calculate_CAE_writeback_cycles(out_feature_size)*2
                print("layer %d's write back cycle count is %d" % (layer_index, layer_write_back_cycle))
                layer_cae_memory_cycle = layer_read_cycle + layer_write_back_cycle
                
                layer_cae_cycle = max(layer_cae_compute_cycle, layer_cae_memory_cycle)

                self.forward_nme_cycles.append(layer_nme_cycle)
                self.forward_cae_cycles.append(layer_cae_cycle)

        # pipeline forward process
        forward_cycles = 0
        if self.change_order and self.input_feature_size > 256: # change order
            forward_cycles = max(self.forward_nme_cycles[0][0], self.forward_cae_cycles[0][0]) + \
                                max(self.forward_nme_cycles[0][1], self.forward_cae_cycles[0][1])
            for layer_index in range(1, self.n_layers):
                forward_cycles += max(self.forward_nme_cycles[layer_index], self.forward_cae_cycles[layer_index])
        else: # not change
            for layer_index in range(self.n_layers):
                forward_cycles += max(self.forward_nme_cycles[layer_index], self.forward_cae_cycles[layer_index])
        return forward_cycles

    def calculate_backward_cycles(self):
        self.backward_nme_cycles = []
        self.backward_cae_cycles = []
        for layer_index in range(self.n_layers-1, -1, -1):
            last_layer = False
            if layer_index == 0:
                # first layer
                in_feature_size = self.hidden_size
                out_feature_size = self.input_feature_size
            elif layer_index == self.n_layers - 1:
                last_layer = True
                in_feature_size = self.n_labels
                out_feature_size = self.hidden_size
            else:
                in_feature_size = self.hidden_size
                out_feature_size = self.hidden_size

            print("layer %d's backward simulation starts" % layer_index)
            if last_layer:
                # last layer's backward do not need to reduce
                layer_nme_cycle = 0
                print("layer %d's nme cycle count is %d" % (layer_index, layer_nme_cycle))

                # CAE compute: backward -> weight update
                layer_cae_backward_cycle = self.calculate_CAE_backward_cycles(in_feature_size, out_feature_size, last_layer)
                print("layer %d's backward cycle count is %d" % (layer_index, layer_cae_backward_cycle))
                layer_cae_update_weight_cycle = self.calculate_CAE_weight_update_cycles(out_feature_size)
                print("layer %d's weight update cycle count is %d" % (layer_index, layer_cae_update_weight_cycle))
                layer_cae_compute_cycle = layer_cae_backward_cycle + layer_cae_update_weight_cycle
                # CAE memory: write back delta
                layer_cae_memory_cycle = self.calculate_CAE_writeback_cycles(out_feature_size)
                print("layer %d's write back cycle count is %d" % (layer_index, layer_cae_memory_cycle))
                # pipeline compute & memory
                layer_cae_cycle = max(layer_cae_compute_cycle, layer_cae_memory_cycle)
            else:
                layer_nme_cycle = self.calculate_NME_reduce_cycles(in_feature_size)
                print("layer %d's nme cycle count is %d" % (layer_index, layer_nme_cycle))

                # CAE compute: merge -> backward -> update weight
                layer_cae_merge_cycle = self.calculate_CAE_merge_cycles(in_feature_size)
                print("layer %d's cae merge cycle count is %d" % (layer_index, layer_cae_merge_cycle))
                layer_cae_backward_cycle = self.calculate_CAE_backward_cycles(in_feature_size, out_feature_size, last_layer)
                print("layer %d's backward cycle count is %d" % (layer_index, layer_cae_backward_cycle))
                layer_cae_update_weight_cycle = self.calculate_CAE_weight_update_cycles(out_feature_size)
                print("layer %d's weight update cycle count is %d" % (layer_index, layer_cae_update_weight_cycle))
                layer_cae_compute_cycle = layer_cae_merge_cycle + layer_cae_backward_cycle + layer_cae_update_weight_cycle
                # CAE memory: read partial sum, write back delta
                layer_cae_memory_cycle = self.calculate_CAE_read_cycles(in_feature_size) + self.calculate_CAE_writeback_cycles(out_feature_size)
                print("layer %d's read & write back cycle count is %d" % (layer_index, layer_cae_memory_cycle))
                # pipeline compute & memory
                layer_cae_cycle = max(layer_cae_compute_cycle, layer_cae_memory_cycle)

            self.backward_nme_cycles.append(layer_nme_cycle)
            self.backward_cae_cycles.append(layer_cae_cycle)

        # pipeline backward process
        # last layer, only CAE works, then NME-CAE pipeline
        backward_cycles = self.backward_cae_cycles[-1]
        for layer_index in range(self.n_layers-2, -1, -1):
            backward_cycles += max(self.backward_nme_cycles[layer_index], self.backward_cae_cycles[layer_index])
        return backward_cycles

    def calculate_cycles(self):
        print("Start calculation:")
        print("Total shards: ", self.total_shards)

        forward_cycles = self.calculate_forward_cycles()
        backward_cycles = self.calculate_backward_cycles()
        total_cycles = forward_cycles + backward_cycles
        print("forward cycle count is: %d" % forward_cycles)
        print("backward cycle count is: %d" % backward_cycles)
        print("total cycle count is: %d" % total_cycles)
        return total_cycles


if __name__ == '__main__':
    dataset = Dataset("NELL")
    system = HGPGCNear()
    system.assign_shards(dataset)
    system.calculate_cycles()