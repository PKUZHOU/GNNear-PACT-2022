#include "cpu.h"

namespace dramsim3 {

void RandomCPU::ClockTick() {
    // Create random CPU requests at full speed
    // this is useful to exploit the parallelism of a DRAM protocol
    // and is also immune to address mapping and scheduling policies
    memory_system_.ClockTick();
    if (get_next_) {
        last_addr_ = gen();
        last_write_ = (gen() % 3 == 0);
    }
    get_next_ = memory_system_.WillAcceptTransaction(last_addr_, last_write_);
    if (get_next_) {
        memory_system_.AddTransaction(last_addr_, last_write_);
    }
    clk_++;
    return;
}

void StreamCPU::ClockTick() {
    // stream-add, read 2 arrays, add them up to the third array
    // this is a very simple approximate but should be able to produce
    // enough buffer hits

    // moving on to next set of arrays
    memory_system_.ClockTick();
    if (offset_ >= array_size_ || clk_ == 0) {
        addr_a_ = gen();
        addr_b_ = gen();
        addr_c_ = gen();
        offset_ = 0;
    }

    if (!inserted_a_ &&
        memory_system_.WillAcceptTransaction(addr_a_ + offset_, false)) {
        memory_system_.AddTransaction(addr_a_ + offset_, false);
        inserted_a_ = true;
    }
    if (!inserted_b_ &&
        memory_system_.WillAcceptTransaction(addr_b_ + offset_, false)) {
        memory_system_.AddTransaction(addr_b_ + offset_, false);
        inserted_b_ = true;
    }
    if (!inserted_c_ &&
        memory_system_.WillAcceptTransaction(addr_c_ + offset_, true)) {
        memory_system_.AddTransaction(addr_c_ + offset_, true);
        inserted_c_ = true;
    }
    // moving on to next element
    if (inserted_a_ && inserted_b_ && inserted_c_) {
        offset_ += stride_;
        inserted_a_ = false;
        inserted_b_ = false;
        inserted_c_ = false;
    }
    clk_++;
    return;
}

TraceBasedCPU::TraceBasedCPU(const std::string& config_file,
                             const std::string& output_dir,
                             const std::string& trace_file)
    : CPU(config_file, output_dir) {
    trace_file_.open(trace_file);
    if (trace_file_.fail()) {
        std::cerr << "Trace file does not exist" << std::endl;
        AbruptExit(__FILE__, __LINE__);
    }
}

void TraceBasedCPU::ClockTick() {
    memory_system_.ClockTick();
    if (!trace_file_.eof()) {
        if (get_next_) {
            get_next_ = false;
            trace_file_ >> trans_;
        }
        if (trans_.added_cycle <= clk_) {
            get_next_ = memory_system_.WillAcceptTransaction(trans_.addr,
                                                             trans_.is_write);
            if (get_next_) {
                memory_system_.AddTransaction(trans_.addr, trans_.is_write);
            }
        }
    }
    clk_++;
    return;
}



GCNearNME::GCNearNME(const std::string& config_file, const std::string& output_dir,
                     std::vector<uint64_t>& DIMM_node_idx, std::vector<std::vector<uint64_t>>& sub_adj_matrix)
    : CPU(config_file, output_dir), config_(new Config(config_file, output_dir)), DIMM_node_idx(DIMM_node_idx), sub_adj_matrix(sub_adj_matrix)
{
    DIMM_node_num = DIMM_node_idx.size();
    DIMM_shard_num = std::ceil(GRAPH_NODE_NUM*1.0/SHARD_SIZE);

    burst_bytes = config_->bus_width / 8 * config_->BL;
    num_type_row_bit = LogBase2(VECTOR_TYPE_NUM);
    num_vector_column_bit = LogBase2(WHOLE_VECTOR_DIM*DIM_BYTES);
    vector_column_mask = (1<<num_vector_column_bit) - 1;

    // bit positions for addr mapping
    num_vertex_column_bit = LogBase2(config_->columns) - num_vector_column_bit;
    num_vertex_bank_bit = LogBase2(config_->ba_mask+1) ;
    num_vertex_bank_group_bit = LogBase2(config_->bg_mask+1);
    num_vertex_channel_bit = LogBase2(CHANNEL_NUM);
    num_vertex_DIMM_bit = LogBase2(DIMM_NUM_PER_CHANNEL);
    num_vertex_row_bit = LogBase2(config_->rows) - num_type_row_bit;

    // shift bit numbers and masks for fetching addr
    vertex_row_pos = num_vertex_DIMM_bit + num_vertex_channel_bit + num_vertex_bank_group_bit + num_vertex_bank_bit + num_vertex_bank_bit + num_vertex_column_bit;
    vertex_row_mask = (1<<num_vertex_row_bit) - 1;
    vertex_bank_group_pos = num_vertex_bank_bit + num_vertex_column_bit;
    vertex_bank_group_mask = (1<<num_vertex_bank_group_bit) - 1;
    vertex_bank_pos = num_vertex_column_bit;
    vertex_bank_mask = (1<<num_vertex_bank_bit) - 1;
    vertex_column_mask = (1<<num_vertex_column_bit) - 1;

    now_shard = 0;
    now_node = 0;
    scan_pos.resize(DIMM_node_num);
    state = NMEState::REDUCE;

    last_sync_shard = finished_shard = -1;
    shard_start_cycle.clear(); shard_start_cycle.emplace_back(clk_);
    shard_work_cycle.clear(); shard_work_cycle.emplace_back(0);
    shard_wait_cycle.clear(); shard_wait_cycle.emplace_back(0);
    shard_states.clear(); shard_states[now_shard] = ShardState(now_shard);
}

void GCNearNME::CheckBound()
{
    // check whether we have finished sending all insts in temporary shard
    // if we have reached the sliding window's boundary, we stop sending insts
    if(now_node==DIMM_node_num)
    {
        if(shard_states[now_shard].all_node_num==0)
            shard_states[now_shard].is_finished = true;
        now_node = 0;
        now_shard++;
        shard_start_cycle.emplace_back(clk_);
        shard_work_cycle.emplace_back(0);
        shard_wait_cycle.emplace_back(0);
        shard_states[now_shard] = ShardState(now_shard);
        //if(now_shard-last_sync_shard > SLIDE_WINDOW_SIZE)
            //return;
            //state = NMEState::SHARD_END;
        //return;
    }

    // update finished shard id
    // note that we agree inner-window scheduling, so we find the last consecutive finished shards
    int now_finished_shard=finished_shard+1;
    for(; now_finished_shard<now_shard; ++now_finished_shard)
    {
        if(!shard_states[now_finished_shard].is_finished)
            break;
    }
    finished_shard = now_finished_shard-1;

    // finally, check whether we have finished all shards
    if(now_shard==DIMM_shard_num)
    {
        //now_node = 0;
        //now_shard = 0;
        //scan_pos.clear(); scan_pos.resize(DIMM_node_num);
        state = NMEState::REDUCE_END;
        return;
    }
}

void GCNearNME::AddTransaction()
{
    while(!undo_read_addrs.empty() && memory_system_.WillAcceptTransaction(undo_read_addrs.front(), false))
    {
        uint64_t addr = undo_read_addrs.front();
        Transaction trans = Transaction(addr, false);
        trans.shard_id = now_shard; trans.node_id = DIMM_node_idx[now_node]; trans.in_node_id = in_node_ids.front();
        memory_system_.AddTransaction(trans);
        undo_read_addrs.pop();
        in_node_ids.pop();
    }

    if(undo_read_addrs.empty())
        now_node++;
}

void GCNearNME::Reduce()
{
    // skip empty shard rows or finished rows
    while((scan_pos[now_node]<sub_adj_matrix[now_node].size() && sub_adj_matrix[now_node][scan_pos[now_node]]>=(now_shard+1)*SHARD_SIZE)
            || scan_pos[now_node]==sub_adj_matrix[now_node].size())
    {
        now_node++;
        CheckBound();
        if(state == NMEState::REDUCE_END || now_shard-last_sync_shard > SLIDE_WINDOW_SIZE)
            return;
    }

    // record how many times are this node's feature being computed.
    uint64_t calculate_count = 0;
    while(scan_pos[now_node]<sub_adj_matrix[now_node].size() && sub_adj_matrix[now_node][scan_pos[now_node]]<(now_shard+1)*SHARD_SIZE)
    {
        calculate_count++;
        scan_pos[now_node]++;
    }

    // generate DRAM Addrs
    // now we assume that one node's feature only has MAX_VECTOR_DIM
    // simulate rank-level parallel by halving vector bytes
    uint64_t node_idx = DIMM_node_idx[now_node];
    uint64_t inst_num = 0;
    for(uint64_t pos=0; pos<WHOLE_VECTOR_DIM*DIM_BYTES/2; pos+=burst_bytes)
    {
        uint64_t row = (((node_idx>>vertex_row_pos)&vertex_row_mask)<<num_type_row_bit) | (FEATURE_VECTOR);
        //uint channel=0;
        uint64_t rank = 0;
        uint64_t bank = (node_idx>>vertex_bank_pos) & vertex_bank_mask;
        uint64_t bank_group = (node_idx>>vertex_bank_group_pos) & vertex_bank_group_mask;
        uint64_t column = (((node_idx&vertex_column_mask)<<num_vector_column_bit) | (pos&vector_column_mask)) >> LogBase2(config_->BL);
        uint64_t addr = (row<<(config_->ro_pos)) | (rank<<(config_->ra_pos)) | (bank<<(config_->ba_pos)) | (bank_group<<(config_->bg_pos)) | column;
        addr <<= 6;
        undo_read_addrs.push(addr);
        in_node_ids.push(inst_num);
        inst_num++;
    }

    // maintain temporary node's state
    assert(shard_states.find(now_shard)!=shard_states.end());
    shard_states[now_shard].all_nodes[node_idx] = NodeState(node_idx, inst_num);
    shard_states[now_shard].all_node_num++;

    // send as many insts as possible
    AddTransaction();

    CheckBound();
    if(state == NMEState::REDUCE_END || now_shard-last_sync_shard > SLIDE_WINDOW_SIZE)
        return;
}

void GCNearNME::ClockTick()
{
    std::vector<Transaction> finished_trans = memory_system_.GCNearClockTick();
    for(auto trans: finished_trans)
    {
        /*std::cout << trans.shard_id << " , " << trans.node_id << std::endl;
        assert(shard_states.count(trans.shard_id)>0);
        assert(shard_states[trans.shard_id].all_nodes.count(trans.node_id)>0);
        assert(shard_states[trans.shard_id].finished_node_num < shard_states[trans.shard_id].all_node_num);
        assert(!shard_states[trans.shard_id].is_finished);
        assert(shard_states[trans.shard_id].all_nodes[trans.node_id].finished_inst_num < 
                shard_states[trans.shard_id].all_nodes[trans.node_id].all_inst_num);
        assert(!shard_states[trans.shard_id].all_nodes[trans.node_id].is_finished[trans.in_node_id]);*/

        shard_states[trans.shard_id].all_nodes[trans.node_id].is_finished[trans.in_node_id] = true;
        shard_states[trans.shard_id].all_nodes[trans.node_id].finished_inst_num++;
        if(shard_states[trans.shard_id].all_nodes[trans.node_id].finished_inst_num == 
                shard_states[trans.shard_id].all_nodes[trans.node_id].all_inst_num)
            shard_states[trans.shard_id].finished_node_num++;
        if(shard_states[trans.shard_id].finished_node_num == shard_states[trans.shard_id].all_node_num && now_shard>trans.shard_id)
            shard_states[trans.shard_id].is_finished = true;
    }

    switch (state)
    {
    case NMEState::REDUCE:
#ifdef RUN_ONE_CHANNEL
        if(now_shard-last_sync_shard <= SLIDE_WINDOW_SIZE)
        {
#endif
            if(undo_read_addrs.empty())
                Reduce();
            else
            {
                AddTransaction();
                CheckBound();
            }
#ifdef RUN_ONE_CHANNEL
            // after reduce, this DIMM might arrive new shard
            if(now_shard-last_sync_shard <= SLIDE_WINDOW_SIZE)
                shard_work_cycle.back()++;
            else
                shard_wait_cycle.back()++;
        
        }
        else
        {
            shard_wait_cycle.back()++;
            CheckBound();
        }
#endif      
        break;
    //case NMEState::SHARD_END:
    case NMEState::REDUCE_END:
        break;
    default:
        assert(false);
        break;
    }

    //one_round_cycle_count++;
    clk_++;
}



ChannelController::ChannelController(uint channel_id): channel_id(channel_id)
{
    // check whether the channel id is legal 
    assert(channel_id < CHANNEL_NUM);
    all_sub_adj_matrix.resize(DIMM_NUM_PER_CHANNEL);
    all_DIMM_node_idx.resize(DIMM_NUM_PER_CHANNEL);
    all_DIMM_controller.clear();

    finish_shard_id = -1;
    sync_shard_id = -1;
    can_transfer = false;
    is_all_reduce_end = false;
    is_reduce_end.resize(DIMM_NUM_PER_CHANNEL);
    for(int i=0; i<DIMM_NUM_PER_CHANNEL; ++i)
        is_reduce_end[i] = false;   

    // initialize all GCNearNME controllers
    for(int i=0; i<DIMM_NUM_PER_CHANNEL; ++i)
    {
        load_sub_adjacent_matrix(i);

        //uint log_id = (i << LogBase2(CHANNEL_NUM)) | channel_id;
        std::stringstream out_dir;
        out_dir << RESULT_SAVE_DIR << "Channel" << channel_id << "/DIMM" << i << "/";
        std::string cmd = "mkdir -p " + out_dir.str();
        system(cmd.c_str());
        //if(access(out_dir.str().c_str(), 0) == -1)
            //mkdir(out_dir.str().c_str(), S_IRWXU);
            
        all_DIMM_controller.push_back(new GCNearNME(CONFIG_FILE, out_dir.str(), all_DIMM_node_idx[i], all_sub_adj_matrix[i]));
    }

    std::stringstream log_out_path;
    log_out_path << RESULT_SAVE_DIR << "Channel" << channel_id << "/transfer.log";
    log_out = new std::ofstream(log_out_path.str());
}

void ChannelController::load_sub_adjacent_matrix(int DIMM_id)
{
    cnpy::npz_t ori;
    std::map<uint, cnpy::NpyArray> graph_npz;
    std::cout << "load " << GRAPH_NAME << ", channel id: " << channel_id << ", DIMM id: " << DIMM_id << std::endl;

    uint npz_id = (DIMM_id << LogBase2(CHANNEL_NUM)) | channel_id;
    std::stringstream ss;
    ss << GRAPH_FILE_ROOT << "/DIMM" << npz_id << ".npz";
    ori = cnpy::npz_load(ss.str());
    all_DIMM_node_idx[DIMM_id].clear();
    all_sub_adj_matrix[DIMM_id].clear();

    // resort matrix by integer order, not string order
    for(auto& it: ori)
    {
        uint key = std::atoi(it.first.c_str());
        graph_npz[key] = it.second;
    }

    // convert to vector form
    for (auto it: graph_npz)
    {
        all_DIMM_node_idx[DIMM_id].emplace_back(it.first);
        all_sub_adj_matrix[DIMM_id].emplace_back();
        uint64_t* data_ptr = it.second.data<uint64_t>();
        for(int i=0; i<it.second.num_vals; ++i)
        {
            all_sub_adj_matrix[DIMM_id].back().emplace_back(data_ptr[i]);
        }
    }
}

void ChannelController::Run()
{
    // drive all DIMMs to work
    for(int i=0; i<DIMM_NUM_PER_CHANNEL; ++i)
    {
        all_DIMM_controller[i]->ClockTick();
    }

    // update channel controller's state
    // check whether there are shards that all DIMMs have finished
    sync_shard_id = 0x7fffffff;
    for(int i=0; i<DIMM_NUM_PER_CHANNEL; ++i)
    {
        sync_shard_id = std::min(sync_shard_id, all_DIMM_controller[i]->finished_shard);
    }
    if(sync_shard_id > finish_shard_id)
        can_transfer = true;
    // check whether all DIMMs have finished all shards
    for(int i=0; i<DIMM_NUM_PER_CHANNEL; ++i)
    {
        if(all_DIMM_controller[i]->GetState() == NMEState::REDUCE_END)
        {
            is_reduce_end[i] = true;
        }
    }
    if(std::count(is_reduce_end.begin(), is_reduce_end.end(), true) == DIMM_NUM_PER_CHANNEL)
        is_all_reduce_end = true;

    // check whether all DIMMs finish the shard they need to deal with
    if(can_transfer)
    {
        //std:: cout << "Shard " << finished_shard_num << " has finished by all DIMMs." << std::endl;
        ScheduleOutput();
    }

    // check whether all DIMMs finish all shards --- this means we have finished the last shard
    if(is_all_reduce_end)
    {
        std::cout << "All DIMMs have finished reduce." << std::endl;
        return;
    }

    //if((finish_shard_id+1) % 100 == 0)
        //std::cout << finish_shard_id << " shards have finished, total shard number is " << std::ceil(GRAPH_NODE_NUM*1.0/SHARD_SIZE) << std::endl;
}

void ChannelController::ScheduleOutput()
{
    // first deal with the order for transfer output
    std::vector<uint64_t> all_DIMM_shard_start_cycle;
    std::vector<uint64_t> all_DIMM_shard_start_work_cycle; // should be the same when SLIDE_WINDOW_SIZE is 1
    std::vector<uint> all_DIMM_shard_finish_work_cycle;
    for(uint shard=finish_shard_id+1; shard<=sync_shard_id; ++shard)
    {
        for(int i=0; i<DIMM_NUM_PER_CHANNEL; ++i)
        {
            uint64_t start_cycle = all_DIMM_controller[i]->shard_start_cycle.front();
            uint64_t wait_cycle = all_DIMM_controller[i]->shard_wait_cycle.front();
            uint64_t work_cycle = all_DIMM_controller[i]->shard_work_cycle.front();
            all_DIMM_shard_start_cycle.emplace_back(start_cycle);
            all_DIMM_shard_start_work_cycle.emplace_back(start_cycle + wait_cycle);
            all_DIMM_shard_finish_work_cycle.emplace_back(start_cycle+ wait_cycle + work_cycle);
            all_DIMM_controller[i]->shard_start_cycle.pop_front();
            all_DIMM_controller[i]->shard_wait_cycle.pop_front();
            all_DIMM_controller[i]->shard_work_cycle.pop_front();
        }

        // check correctness
        // if(SLIDE_WINDOW_SIZE==1)
        // {
        //     assert(std::count(all_DIMM_shard_start_work_cycle.begin(), all_DIMM_shard_start_work_cycle.end(), all_DIMM_shard_start_work_cycle[0]) == DIMM_NUM_PER_CHANNEL);

        //     if(all_DIMM_controller[0]->now_shard == 135)
        //     {

        //     }
        // }

        uint64_t max_start_wait_cycle = *std::max_element(all_DIMM_shard_start_cycle.begin(), all_DIMM_shard_start_cycle.end());
        uint64_t min_start_wait_cycle = *std::min_element(all_DIMM_shard_start_cycle.begin(), all_DIMM_shard_start_cycle.end());
        uint64_t max_start_work_cycle = *std::max_element(all_DIMM_shard_start_work_cycle.begin(), all_DIMM_shard_start_work_cycle.end());
        uint64_t min_start_work_cycle = *std::min_element(all_DIMM_shard_start_work_cycle.begin(), all_DIMM_shard_start_work_cycle.end());
        uint64_t max_finish_cycle = *std::max_element(all_DIMM_shard_finish_work_cycle.begin(), all_DIMM_shard_finish_work_cycle.end());
        uint64_t min_finish_cycle = *std::min_element(all_DIMM_shard_finish_work_cycle.begin(), all_DIMM_shard_finish_work_cycle.end());
        (*log_out) << "shard id: " << shard;
        (*log_out) << ", min start wait cycle is: " << min_start_wait_cycle << ", max start wait cycle is: " << max_start_wait_cycle;
        (*log_out) << ", min start work cycle is: " << min_start_work_cycle << ", max start work cycle is: " << max_start_work_cycle;
        (*log_out) << ", min finish work cycle is: " << min_finish_cycle << ", max finish work cycle is: " << max_finish_cycle << std::endl; 
    }


    // then update all DIMM's state and reset channel controller's state
    can_transfer = false;
    finish_shard_id = sync_shard_id;
    for(int i=0; i<DIMM_NUM_PER_CHANNEL; ++i)
    {
        all_DIMM_controller[i]->last_sync_shard = finish_shard_id;
    }
}

void ChannelController::LogResult()
{
    for(int i=0; i<DIMM_NUM_PER_CHANNEL; ++i)
    {
        all_DIMM_controller[i]->PrintStats();
    }
}

ChannelController::~ChannelController()
{
    for(int i=0; i<DIMM_NUM_PER_CHANNEL; ++i)
    {
        delete all_DIMM_controller[i];
    }
    log_out->close();
}


}  // namespace dramsim3
