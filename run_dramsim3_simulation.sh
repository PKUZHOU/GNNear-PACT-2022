#!/bin/bash

PROJECT_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd );


# build dynamic library of cnpy
cd ${PROJECT_DIR}/dramsim3_simulation/ext/cnpy
cnpy_folder=${PROJECT_DIR}/dramsim3_simulation/ext/cnpy/install
if [ -d "$cnpy_folder" ]; then
    echo "cnpy has already been built"
else
    mkdir build
    mkdir install
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=../install ../
    make
    make install
fi

# generate graph partitions
ogb_folder=${PROJECT_DIR}/dramsim3_simulation/graph_partition/Amazon/;
pyg_folder=${PROJECT_DIR}/dramsim3_simulation/graph_partition/Reddit/;
cd ${PROJECT_DIR}/dramsim3_simulation/graph_partition
if [ -d "$ogb_folder" ]; then
    echo "Amazon data has already generated"
else
    python ogb_partition.py
fi 
if [ -d "$pyg_folder" ]; then
    echo "Reddit data has already generated"
else
    python pyg_partition.py
fi

# build main projects
cd ${PROJECT_DIR}/dramsim3_simulation/
mkdir -p build
cd build
cmake ..
make -j4
mv dramsim3main ../

# run main program
cd ${PROJECT_DIR}/dramsim3_simulation/
./dramsim3main
mv ./results/* ../results
rm -rf results
