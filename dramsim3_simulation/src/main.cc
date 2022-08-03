#include <iostream>
#include <string>
#include <map>

#include <sys/io.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

#include <sys/time.h>

#include "./../ext/headers/args.hxx"
#include "cpu.h"

using namespace dramsim3;

int main(int argc, const char **argv) {
#ifdef RUN_ORI_SIM
    args::ArgumentParser parser(
        "DRAM Simulator.",
        "Examples: \n."
        "./build/dramsim3main configs/DDR4_8Gb_x8_3200.ini -c 100 -t "
        "sample_trace.txt\n"
        "./build/dramsim3main configs/DDR4_8Gb_x8_3200.ini -s random -c 100");
    args::HelpFlag help(parser, "help", "Display the help menu", {'h', "help"});
    args::ValueFlag<uint64_t> num_cycles_arg(parser, "num_cycles",
                                             "Number of cycles to simulate",
                                             {'c', "cycles"}, 100000);
    args::ValueFlag<std::string> output_dir_arg(
        parser, "output_dir", "Output directory for stats files",
        {'o', "output-dir"}, ".");
    args::ValueFlag<std::string> stream_arg(
        parser, "stream_type", "address stream generator - (random), stream",
        {'s', "stream"}, "");
    args::ValueFlag<std::string> trace_file_arg(
        parser, "trace",
        "Trace file, setting this option will ignore -s option",
        {'t', "trace"});
    args::Positional<std::string> config_arg(
        parser, "config", "The config file name (mandatory)");

    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    std::string config_file = args::get(config_arg);
    if (config_file.empty()) {
        std::cerr << parser;
        return 1;
    }

    uint64_t cycles = args::get(num_cycles_arg);
    std::string output_dir = args::get(output_dir_arg);
    std::string trace_file = args::get(trace_file_arg);
    std::string stream_type = args::get(stream_arg);

    CPU *cpu;
    if (!trace_file.empty()) {
        cpu = new TraceBasedCPU(config_file, output_dir, trace_file);
    } else {
        if (stream_type == "stream" || stream_type == "s") {
            cpu = new StreamCPU(config_file, output_dir);
        } else {
            cpu = new RandomCPU(config_file, output_dir);
        }
    }

    if(!trace_file.empty())
    {
        while(!((TraceBasedCPU*)cpu)->IsFileEnd())
        {
            cpu->ClockTick();
        }
    }
    else
    {
        for (uint64_t clk = 0; clk < cycles; clk++) 
        {
            cpu->ClockTick();
        }
    }
    
    cpu->PrintStats();

    delete cpu;

#elif defined(RUN_ONE_CHANNEL)

    struct timeval start;
    gettimeofday(&start, NULL);

    ChannelController* channel_controller = new ChannelController(CHANNEL_ID);
    while(!channel_controller->FinishAllReduce())
    {
        channel_controller->Run();
    }
    channel_controller->LogResult();
    delete channel_controller;

    struct timeval end;
    gettimeofday(&end, NULL);
    double delta = (end.tv_sec + end.tv_usec*0.000001) - (start.tv_sec + start.tv_usec*0.000001);
    std::cout << "simulate time is: " << delta << "seconds" << std::endl;

#else
    std::cout << "Please choose one mode to simulate!" << std::endl;
#endif // RUN_ORI_SIM
    return 0;
}
