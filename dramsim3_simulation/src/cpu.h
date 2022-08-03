#ifndef __CPU_H
#define __CPU_H

#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <deque>
#include <vector>
#include <queue>
#include <algorithm>
#include "common.h"
#include "memory_system.h"
#include "graph_settings.h"

#include <sys/io.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

namespace dramsim3 {

class CPU {
   public:
    CPU(const std::string& config_file, const std::string& output_dir)
        : memory_system_(
              config_file, output_dir,
              std::bind(&CPU::ReadCallBack, this, std::placeholders::_1),
              std::bind(&CPU::WriteCallBack, this, std::placeholders::_1)),
          clk_(0) {}
    virtual void ClockTick() = 0;
    void ReadCallBack(uint64_t addr) { return; }
    void WriteCallBack(uint64_t addr) { return; }
    void PrintStats() { memory_system_.PrintStats(); }

   protected:
    MemorySystem memory_system_;
    uint64_t clk_;
};

class RandomCPU : public CPU {
   public:
    using CPU::CPU;
    void ClockTick() override;

   private:
    uint64_t last_addr_;
    bool last_write_ = false;
    std::mt19937_64 gen;
    bool get_next_ = true;
};

class StreamCPU : public CPU {
   public:
    using CPU::CPU;
    void ClockTick() override;

   private:
    uint64_t addr_a_, addr_b_, addr_c_, offset_ = 0;
    std::mt19937_64 gen;
    bool inserted_a_ = false;
    bool inserted_b_ = false;
    bool inserted_c_ = false;
    const uint64_t array_size_ = 2 << 20;  // elements in array
    const int stride_ = 64;                // stride in bytes
};

class TraceBasedCPU : public CPU {
   public:
    TraceBasedCPU(const std::string& config_file, const std::string& output_dir,
                  const std::string& trace_file);
    ~TraceBasedCPU() { trace_file_.close(); }
    void ClockTick() override;

    bool IsFileEnd() {return trace_file_.eof();}

   private:
    std::ifstream trace_file_;
    Transaction trans_;
    bool get_next_ = true;
};

















struct NodeState
{
public:
    uint64_t node_id;
    uint64_t all_inst_num;
    uint64_t finished_inst_num;
    std::vector<bool> is_finished;

    NodeState(){}
    NodeState(uint64_t node_id, uint64_t all_inst_num)
        : node_id(node_id), all_inst_num(all_inst_num)
    {
        finished_inst_num = 0;
        is_finished.clear();
        for(int i=0; i<all_inst_num; ++i)
        {
            is_finished.emplace_back(false);
        }
    }
};

struct ShardState
{
public:
    int shard_id;
    uint64_t all_node_num;
    uint64_t finished_node_num;
    std::map<uint64_t, NodeState> all_nodes;
    bool is_finished;

    ShardState(){}
    ShardState(int shard_id): shard_id(shard_id)
    {
        all_node_num = finished_node_num = 0;
        all_nodes.clear();
        is_finished = false;
    }
};

enum class NMEState {REDUCE, /*SHARD_END,*/ REDUCE_END};

class GCNearNME: public CPU
{
public:
    GCNearNME(const std::string& config_file, const std::string& output_dir,
              std::vector<uint64_t>& DIMM_node_idx, std::vector<std::vector<uint64_t>>& sub_adj_matrix);
    void ClockTick() override;

    NMEState GetState(){return state;}
    void SetState(NMEState newstate){state = newstate;}


    uint64_t now_node;
    std::queue<uint64_t> undo_read_addrs;
    std::queue<uint64_t> in_node_ids;

    int now_shard;
    int finished_shard;
    int last_sync_shard;
    std::deque<uint64_t> shard_start_cycle;
    std::deque<uint64_t> shard_work_cycle;
    std::deque<uint64_t> shard_wait_cycle;
    std::map<int, ShardState> shard_states;

private:
    void Reduce();
    void AddTransaction();
    void CheckBound();

    Config* config_;
    uint64_t DIMM_node_num; // how many nodes are assigned to this DIMM
    std::vector<uint64_t>& DIMM_node_idx; // node indexes that are assigned to this DIMM
    std::vector<std::vector<uint64_t>>& sub_adj_matrix; // (compressed) adjacent vectors of every node
    uint64_t DIMM_shard_num; // how many shards are split.

    int burst_bytes;
    int num_type_row_bit;
    int num_vector_column_bit;
    int num_vertex_column_bit;
    int num_vertex_bank_bit;
    int num_vertex_bank_group_bit;
    int num_vertex_channel_bit;
    int num_vertex_DIMM_bit;
    int num_vertex_row_bit;

    int vertex_row_pos;
    uint vertex_row_mask;
    int vertex_bank_pos;
    uint vertex_bank_mask;
    int vertex_bank_group_pos;
    uint vertex_bank_group_mask;
    uint vertex_column_mask;
    uint vector_column_mask;

    std::vector<uint64_t> scan_pos;
    NMEState state;
};

class ChannelController
{
    std::ofstream* log_out;

    uint channel_id;
    std::vector<std::vector<uint64_t>> all_DIMM_node_idx;
    std::vector<std::vector<std::vector<uint64_t>>> all_sub_adj_matrix;
    std::vector<GCNearNME*> all_DIMM_controller;

    int finish_shard_id;
    int sync_shard_id;
    bool can_transfer;
    bool is_all_reduce_end;
    std::vector<bool> is_reduce_end;

public:
    ChannelController(uint channel_id);
    ~ChannelController();

    bool FinishAllReduce() {return is_all_reduce_end;}

    void load_sub_adjacent_matrix(int DIMM_id);

    void Run();

    void ScheduleOutput();

    void LogResult();
};

}  // namespace dramsim3
#endif
