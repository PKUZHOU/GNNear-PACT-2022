import multiprocessing
import os
import argparse
from GNNear_simulator import *
from multiprocessing import  Process
import numpy as np
import matplotlib.pyplot as plt


def rank_exploration_mp(args):
    print('------Rank Number Exploration------')
    all_dataset = ['RD']

    def rank_eval(rank_num, dataset_name, dataset_root_dir, queue):
        dataset = Dataset(dataset_name, dataset_root_path=dataset_root_dir)
        system = HGPGCNear(rank_num=rank_num)
        system.assign_shards(dataset)
        cycle_cnt = system.calculate_cycles()
        queue.put((rank_num, cycle_cnt))

    all_result_dict = {}
    for dataset in all_dataset:
        result_queue = multiprocessing.Queue()
        rank_num = 1
        process_list = []
        while rank_num < 9:
            p = Process(target=rank_eval,args=(rank_num, dataset, args.dataset_root_dir, result_queue))
            p.start()
            process_list.append(p)
            rank_num += 1

        result_dict = {}
        for p in process_list:
            p.join()
            result_tuple = result_queue.get()
            result_dict[result_tuple[0]] = result_tuple[1]
        all_result_dict[dataset] = result_dict
        
        f = open("results"+"/"+ "rank/" + dataset+".log", "a+")
        rank_num = 1
        while rank_num < 9:
            print("rank number %d, Sppedup over rank number = 1 is %.2f" % (rank_num, result_dict[1]/result_dict[rank_num]))
            f.write("rank number %d, Sppedup over rank number = 1 is %.2f\n" % (rank_num, result_dict[1]/result_dict[rank_num]))
            rank_num += 1
        f.flush()
        f.close()

    plt.figure(figsize=(10, 8))
    x = [i for i in range(1, 9)]
    x_ticks = [i for i in range(1, 9)]
    plt.xticks(x, x_ticks)
    plt.ylim(0, 4)
    for dataset in all_dataset:
        result_list = []
        rank_num = 1
        while rank_num < 9:
            result_list.append(all_result_dict[dataset][rank_num])
            rank_num += 1
        relative_result = [result_list[0]/result_list[i] for i in range(len(result_list))]
        plt.bar(x, relative_result, label=dataset)
    plt.savefig('./results/rank/rank.png')
    plt.close()

    print('--Rank Number Exploration Finished--')


def ratio_exploration_mp(args):
    print('------Duplication Ratio Exploration------')
    all_dataset = ['PT', 'RD', 'YP', 'AM']

    def ratio_eval(ratio, dataset_name, dataset_root_dir, queue):
        dataset = Dataset(dataset_name, dataset_root_path=dataset_root_dir)
        system = HGPGCNear(dup_ratio=ratio)
        system.assign_shards(dataset)
        cycle_cnt = system.calculate_cycles()
        queue.put((ratio, cycle_cnt))
    
    all_result_dict = {}
    for dataset in all_dataset:
        result_queue = multiprocessing.Queue()
        ratio = 0.0
        process_list = []
        while ratio < 0.6:
            p = Process(target=ratio_eval,args=(ratio, dataset, args.dataset_root_dir, result_queue))
            p.start()
            process_list.append(p)
            ratio += 0.1

        result_dict = {}
        for p in process_list:
            p.join()
            result_tuple = result_queue.get()
            result_dict[result_tuple[0]] = result_tuple[1]
        all_result_dict[dataset] = result_dict
        
        f = open("results"+"/"+ "ratio/" + dataset+".log", "a+")
        ratio = 0.0
        while ratio < 0.6:
            print("duplication ratio %.2f, Sppedup over duplication ratio = 0.0 is %.2f" % (ratio, result_dict[0.0]/result_dict[ratio]))
            f.write("duplication ratio %.2f, Sppedup over duplication ratio = 0.0 is %.2f\n" % (ratio, result_dict[0.0]/result_dict[ratio]))
            ratio += 0.1
        f.flush()
        f.close()

    plt.figure(figsize=(10, 8))
    x = [i for i in range(6)]
    x_ticks = [('%.2f' % (0.1*i)) for i in range(6)]
    plt.xticks(x, x_ticks)
    plt.ylim(0.6, 1.6)
    for dataset in all_dataset:
        result_list = []
        ratio = 0.0
        while ratio < 0.6:
            result_list.append(all_result_dict[dataset][ratio])
            ratio += 0.1
        relative_result = [result_list[0]/result_list[i] for i in range(len(result_list))]
        plt.plot(x, relative_result, label=dataset)
    plt.legend()
    plt.savefig('./results/ratio/ratio.png')
    plt.close()

    print('--Duplication Ratio Exploration Finished--')


def shard_size_exploration_mp(args):
    print('------Shard Size Exploration------')
    all_dataset = ['PT', 'RD', 'YP', 'AM']

    def shard_size_eval(shard_size, dataset_name, dataset_root_dir, queue):
        dataset = Dataset(dataset_name, dataset_root_path=dataset_root_dir)
        system = HGPGCNear(interval=shard_size)
        system.assign_shards(dataset)
        cycle_cnt = system.calculate_cycles()
        queue.put((shard_size, cycle_cnt))
        
    all_result_dict = {}
    for dataset in all_dataset:
        result_queue = multiprocessing.Queue()
        shard_size = 1
        process_list = []
        while shard_size < 512:
            p = Process(target=shard_size_eval,args=(shard_size, dataset, args.dataset_root_dir, result_queue))
            p.start()
            process_list.append(p)
            shard_size *= 2

        result_dict = {}
        for p in process_list:
            p.join()
            result_tuple = result_queue.get()
            result_dict[result_tuple[0]] = result_tuple[1]
        all_result_dict[dataset] = result_dict
        
        f = open("results"+"/"+ "shard/" + dataset+".log", "a+")
        shard_size = 1
        while shard_size < 512:
            print("shard size %d, Sppedup over shard size = 1 is %.2f" % (shard_size, result_dict[1]/result_dict[shard_size]))
            f.write("shard size %d, Sppedup over shard size = 1 is %.2f\n" % (shard_size, result_dict[1]/result_dict[shard_size]))
            shard_size *= 2
        f.flush()
        f.close()
    
    plt.figure(figsize=(10, 8))
    x = [i for i in range(9)]
    x_ticks = [2**i for i in range(9)]
    plt.xticks(x, x_ticks)
    plt.ylim(0.9, 3.9)
    for dataset in all_dataset:
        result_list = []
        shard_size = 1
        while shard_size < 512:
            result_list.append(all_result_dict[dataset][shard_size])
            shard_size *= 2
        relative_result = [result_list[0]/result_list[i] for i in range(len(result_list))]
        plt.plot(x, relative_result, label=dataset)
    plt.legend()
    plt.savefig('./results/shard/shard.png')
    plt.close()

    print('--Shard Size Exploration Finished--')


def window_size_exploreation_mp(args):
    print('------Window Size Exploration------')
    all_dataset = ['PT', 'RD', 'YP', 'AM']

    def window_size_eval(window_size, dataset_name, dataset_root_dir, queue):
        dataset = Dataset(dataset_name, dataset_root_path=dataset_root_dir)
        system = HGPGCNear(window_size=window_size)
        system.assign_shards(dataset)
        cycle_cnt = system.calculate_cycles()
        queue.put((window_size, cycle_cnt))

    all_result_dict = {}
    for dataset in all_dataset:
        process_list = []
        result_queue = multiprocessing.Queue()
        window_size = 1
        while window_size < 64:
            p = Process(target=window_size_eval,args=(window_size, dataset, args.dataset_root_dir, result_queue))
            p.start()
            process_list.append(p)
            window_size *= 2
        
        result_dict = {}
        for p in process_list:
            p.join()
            result_tuple = result_queue.get()
            result_dict[result_tuple[0]] = result_tuple[1]
        all_result_dict[dataset] = result_dict
        
        f = open("results"+"/"+ "window/" + dataset+".log", "a+")
        window_size = 1
        while window_size < 64:
            print("window size %d, Sppedup over window size = 1 is %.2f" % (window_size, result_dict[1]/result_dict[window_size]))
            f.write("window size %d, Sppedup over window size = 1 is %.2f\n" % (window_size, result_dict[1]/result_dict[window_size]))
            window_size *= 2
        f.flush()
        f.close()

    plt.figure(figsize=(10, 8))
    x = [i for i in range(6)]
    x_ticks = [2**i for i in range(6)]
    plt.xticks(x, x_ticks)
    plt.ylim(0.9, 3.3)
    for dataset in all_dataset:
        result_list = []
        window_size = 1
        while window_size < 64:
            result_list.append(all_result_dict[dataset][window_size])
            window_size *= 2
        relative_result = [result_list[0]/result_list[i] for i in range(len(result_list))]
        plt.plot(x, relative_result, label=dataset)
    plt.legend()
    plt.savefig('./results/window/window.png')
    plt.close()

    print('--Window Size Exploration Finished--')


def ieo_comparison(args):
    print('------IEO Comparison------')
    system_settings = [
        {"window_size":4, "interval":128, "ratio":0.0, "inter_schedule":True, "use_NDP":True, 'change_order':False, 'type':'No IEO'}, # no ieo
        {"window_size":4, "interval":128, "ratio":0.0, "inter_schedule":True, "use_NDP":True, 'change_order':True, 'type':'Apply IEO'},   # use ieo
    ]

    def simulate(setting, dataset_name, dataset_root_dir, que):
        dataset = Dataset(dataset_name, dataset_root_path=dataset_root_dir)
        system = HGPGCNear(window_size=setting["window_size"], interval=setting["interval"], dup_ratio=setting["ratio"], 
                            inter_schedule=setting["inter_schedule"], use_NDP=setting["use_NDP"], change_order=setting['change_order'])
        system.assign_shards(dataset)
        cycle_cnt = system.calculate_cycles()
        que.put((setting['type'], cycle_cnt/standard_frequency))
    
    result_queue = multiprocessing.Queue()
    process_list = []
    for setting in system_settings:
        p = Process(target=simulate, args=(setting, 'RD', args.dataset_root_dir, result_queue))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()
    
    result_dict = {}
    for p in process_list:
        result_pair = result_queue.get()
        result_dict[result_pair[0]] = result_pair[1]
    f = open('./results/ieo/IEO_comparison.log', 'a+')
    no_ieo_performance = result_dict['No IEO']
    ieo_performance = result_dict['Apply IEO']
    print('Reddit dataset, Reduce first time consumption %.2f, Update first time consumption %.2f, speed up ratio %.2f'
            % (no_ieo_performance, ieo_performance, no_ieo_performance/ieo_performance))
    f.write('Reddit dataset, Reduce first time consumption %.2f, Update first time consumption %.2f, speed up ratio %.2f\n'
            % (no_ieo_performance, ieo_performance, no_ieo_performance/ieo_performance))
    f.flush()
    f.close()

    bar_width = 0.3
    x = np.array([1])
    x_label = ['Reddit']
    no_ieo = [no_ieo_performance]
    ieo = [ieo_performance]
    plt.bar(x-bar_width, no_ieo, width=bar_width, label='Reduction First')
    plt.bar(x+bar_width, ieo, width=bar_width, label='Update First')
    plt.ylabel('Sec. Per Epoch')
    plt.ylim(0, 1)
    plt.xticks(x, x_label)
    plt.legend()
    plt.savefig('./results/ieo/IEO_comparison.png')
    plt.close()
    print("--IEO Comparison Finished--")


def speed_up_breakdown_mp(args):
    print('------Speed Up Breakdown------')
    system_settings = [
        {"window_size":1, "interval":1, "ratio":0.0, "inter_schedule":False, "use_NDP":False, 'type':'only CAE'},   # not use NDP
        {"window_size":1, "interval":1, "ratio":0.0, "inter_schedule":False, "use_NDP":True, 'type':'Near-Memory Reduction'},    # use NDP
        {"window_size":1, "interval":128, "ratio":0.0, "inter_schedule":False, "use_NDP":True, 'type':'Narrow Shard'},  # open shard
        {"window_size":4, "interval":128, "ratio":0.35, "inter_schedule":False, "use_NDP":True, 'type':'HGP/Interval Scheduling'},  # duplicate
        {"window_size":4, "interval":128, "ratio":0.35, "inter_schedule":True, "use_NDP":True, 'type':'Inter-Shard Overlapping'},   # inter-shard schedule
    ]

    def simulate(setting, dataset_name, dataset_root_dir, que):
        dataset = Dataset(dataset_name, dataset_root_path=dataset_root_dir)
        system = HGPGCNear(window_size=setting["window_size"], interval=setting["interval"], dup_ratio=setting["ratio"], inter_schedule=setting["inter_schedule"], use_NDP=setting["use_NDP"])
        system.assign_shards(dataset)
        cycle_cnt = system.calculate_cycles()
        que.put((setting['type'], cycle_cnt/standard_frequency))

    result_queue = multiprocessing.Queue()
    process_list = []
    for setting in system_settings:
        p = Process(target=simulate, args=(setting, 'AM', args.dataset_root_dir, result_queue))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()

    result_dict = {}
    for p in process_list:
        result_pair = result_queue.get()
        result_dict[result_pair[0]] = result_pair[1]
    f = open('./results/breakdown/breakdown.log', 'a+')
    CPU_baseline = 17.99
    for i in range(len(system_settings)):
        performance = result_dict[system_settings[i]['type']]
        if i == 0:
            print('CPU baseline time is %.2f' % CPU_baseline)
            print('%s is %.2f times faster than CPU baseline' % (system_settings[i]['type'], CPU_baseline/performance))
            f.write('CPU baseline time is %.2f\n' % CPU_baseline)
            f.write('%s is %.2f times faster than CPU baseline\n' % (system_settings[i]['type'], CPU_baseline/performance))
        else:
            performance_prev = result_dict[system_settings[i-1]['type']]
            print('%s is %.2f times faster than %s' % (system_settings[i]['type'], performance_prev/performance, system_settings[i-1]['type']))
            f.write('%s is %.2f times faster than %s\n' % (system_settings[i]['type'], performance_prev/performance, system_settings[i-1]['type']))
    f.close()

    y_ticks = ['CPU'] + [system_settings[i]['type'] for i in range(len(system_settings))]
    y = range(len(y_ticks))
    relative_results = [1, ]
    for i in range(len(system_settings)):
        performance = result_dict[system_settings[i]['type']]
        relative_results.append(CPU_baseline/performance)
    plt.barh(y, relative_results)
    plt.yticks(y, y_ticks)
    # for i in y:
    #     plt.text(relative_results[i], y[i], s=('%.2f' % relative_results[i]))
    plt.tight_layout()
    plt.savefig('./results/breakdown/breakdown.png')
    plt.close()
    print("--Speed Up Breakdown Finished--")


def throughput_comparison(args):
    print('------Throughput Comparison------')
    all_model = ['GCN', 'GIN', 'SAGEConv', 'GAT']
    all_dataset = ['PT', 'RD', 'YP', 'AM']

    CPU_results = {
        'GCN' : {'PT':1.43, 'RD':7.36, 'YP':3.23, 'AM':17.99},
        'GIN' : {'PT':5.36, 'RD':10.5, 'YP':5.75, 'AM':35},
        'SAGEConv' : {'PT':11.05, 'RD':7.73, 'YP':6.96, 'AM':18.24},
        'GAT' : {'PT':10.18, 'RD':16, 'YP':5.45, 'AM':58}
    }
    GPU_results = {
        'GCN' : {'PT':0.32, 'RD':0.74, 'YP':0.30, 'AM':2.12},
        'GIN' : {'PT':0.38, 'RD':1.05, 'YP':0.35, 'AM':0 }, # AM is OOM
        'SAGEConv' : {'PT':0.35, 'RD':0.76, 'YP':0.33, 'AM':2.188},
        'GAT' : {'PT':0.74, 'RD':1.39, 'YP':0.40, 'AM':3.24}
    }
    GNNear_results = {
        'GCN':None,
        'GIN':None,
        'SAGEConv':None,
        'GAT':None
    }

    def simulate(model, dataset_name, que):
        dataset = Dataset(dataset_name, dataset_root_path=args.dataset_root_dir)
        is_GAT = (model == 'GAT')
        is_GIN = (model == 'GIN')
        if dataset_name == 'PT':
            system = HGPGCNear(dup_ratio=0, is_GAT=is_GAT, is_GIN=is_GIN)
        else:
            system = HGPGCNear(is_GAT=is_GAT, is_GIN=is_GIN)
        
        system.assign_shards(dataset)
        cycle_cnt = system.calculate_cycles()
        que.put((dataset_name, cycle_cnt/standard_frequency))
    
    for model in all_model:
        process_list = []
        model_results = multiprocessing.Queue()
        for dataset in all_dataset:
            p = Process(target=simulate, args=(model, dataset, model_results))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()
        print("Model %s's simulation has finished." % model)
        
        model_results_dict = {}
        for p in process_list:
            result_pair = model_results.get()
            model_results_dict[result_pair[0]] = result_pair[1]
        GNNear_results[model] = model_results_dict

    for model in all_model:
        print('Throughput comparison of model %s:' % model)
        f = open(('./results/throughput/%s.log' % model), 'a+')
        for dataset in all_dataset:
            cpu_time = CPU_results[model][dataset]
            gpu_time = GPU_results[model][dataset]
            gnnear_time = GNNear_results[model][dataset]
            if model == 'GIN' and dataset == 'AM':
                print("\tDataset %s, CPU Sec/Epoch %.3f, Normalized GPU throughput --- (OOM), Normalized GNNear throughput %.2f"
                    % (dataset, cpu_time, cpu_time/gnnear_time))
                f.write("Dataset %s, CPU Sec/Epoch %.3f, Normalized GPU throughput --- (OOM), Normalized GNNear throughput %.2f\n"
                    % (dataset, cpu_time, cpu_time/gnnear_time))
            else:
                print("\tDataset %s, CPU Sec/Epoch %.3f, Normalized GPU throughput %.2f, Normalized GNNear throughput %.2f"
                    % (dataset, cpu_time, cpu_time/gpu_time, cpu_time/gnnear_time))
                f.write("Dataset %s, CPU Sec/Epoch %.3f, Normalized GPU throughput %.2f, Normalized GNNear throughput %.2f\n"
                    % (dataset, cpu_time, cpu_time/gpu_time, cpu_time/gnnear_time))
        f.flush()
        f.close()

    for model in all_model:
        bar_width = 0.3
        x = np.array(range(4))
        fig, ax1 = plt.subplots()
        plt.title(model)
        plt.xticks(x+bar_width, all_dataset)
        ax1.set_ylabel('Norm. Throughput')
        ax1.set_yscale('log')
        # bars
        CPU_perf = list(CPU_results[model].values())
        GPU_perf = list(GPU_results[model].values())
        GNNear_perf = list(GNNear_results[model].values())
        normed_CPU = [1, 1, 1, 1]
        normed_GPU = [(CPU_perf[i]/GPU_perf[i] if GPU_perf[i]!=0 else 0) for i in range(len(all_dataset))]
        normed_GNNear = [CPU_perf[i]/GNNear_perf[i] for i in range(len(all_dataset))]
        ax1.bar(x, normed_CPU, width=bar_width, align='center', label='CPU', alpha=0.5)
        ax1.bar(x+bar_width, normed_GPU, width=bar_width, align='center', label='GPU', alpha=0.5)
        ax1.bar(x+2*bar_width, normed_GNNear, width=bar_width, align='center', label='GNNear', alpha=0.5)
        ax1.legend()
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('CPU Sec/Epoch')
        # ax2.set_yticks([0, 20, 40, 60])
        ax2.plot(x+bar_width, CPU_results[model].values(), marker='^', label = 'DGL-CPU Sec/Epoch')
        ax2.legend()

        plt.savefig('./results/throughput/%s.png' % model)
        plt.close()

    print("--Throughput Comparison Finished--")


def parse_args():
    parser = argparse.ArgumentParser(description='Artifact for experiments in GNNear')
    parser.add_argument('--dataset_root_dir', type=str, default='./dataset/', help='root directory to save all graph datasets')
    
    parser.add_argument('--throughput', action='store_true', help='run Training Throughput Comparison experiments')
    parser.add_argument('--breakdown', action='store_true', help='run Speedup Breakdown experiments')
    parser.add_argument('--ieo', action='store_true', help='run Interchange Execution Order experiments')
    parser.add_argument('--shard', action='store_true', help='run Shard Size Exploration experiments')
    parser.add_argument('--window', action='store_true', help='run Window Size Exploration experiments')
    parser.add_argument('--ratio', action='store_true', help='run Duplication Ratio Exploration experiments')
    parser.add_argument('--rank', action='store_true', help='run Rank Number Exploration experiments')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if args.throughput:
        if not os.path.exists('./results/throughput/'):
            os.mkdir('./results/throughput/')
        throughput_comparison(args)
        return
    elif args.breakdown:
        if not os.path.exists('./results/breakdown'):
            os.mkdir('./results/breakdown')
        speed_up_breakdown_mp(args)
        return
    elif args.ieo:
        if not os.path.exists('./results/ieo'):
            os.mkdir('./results/ieo')
        ieo_comparison(args)
        return
    elif args.shard:
        if not os.path.exists('./results/shard'):
            os.mkdir('./results/shard')
        shard_size_exploration_mp(args)
        return
    elif args.window:
        if not os.path.exists('./results/window'):
            os.mkdir('./results/window')
        window_size_exploreation_mp(args)
        return
    elif args.ratio:
        if not os.path.exists('./results/ratio'):
            os.mkdir('./results/ratio')
        ratio_exploration_mp(args)
        return
    elif args.rank:
        if not os.path.exists('./results/rank'):
            os.mkdir('./results/rank')
        rank_exploration_mp(args)
        return
        

if __name__ == '__main__':
    main()