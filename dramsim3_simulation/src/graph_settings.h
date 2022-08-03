#include <vector>
#include "cnpy.h"

#define CONFIG_FILE "./configs/DDR4_8Gb_x8_2400_2.ini"

#define USE_REDDIT
//#define USE_AMAZON

#ifdef USE_REDDIT
    #define GRAPH_NAME "Reddit"
    #define GRAPH_NODE_NUM 232965
    #define GRAPH_EDGE_NUM 114615892
    #define GRAPH_FILE_ROOT "./graph_partition/Reddit/"
    #define RESULT_SAVE_DIR "./results/Reddit/"
#elif defined(USE_AMAZON)
    #define GRAPH_NAME "Amazon2M"
    #define GRAPH_NODE_NUM 2449029
    #define GRAPH_EDGE_NUM 123718152
    #define GRAPH_FILE_ROOT "./graph_partition/Amazon/"
    #define RESULT_SAVE_DIR "./results/Amazon/"
#endif


//#define RUN_ORI_SIM
//#define RUN_ONE_CHANNEL


#define CHANNEL_NUM 4
#define DIMM_NUM_PER_CHANNEL 4

#define SHARD_SIZE 256
#define SLIDE_WINDOW_SIZE 4

#define VECTOR_TYPE_NUM 8
#define FEATURE_VECTOR 0

#define WHOLE_VECTOR_DIM (256/2)
#define DIM_BYTES 4

#define CHANNEL_ID 0
// #define DIMM_ID 0