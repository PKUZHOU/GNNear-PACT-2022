#!/bin/bash

DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )
cd $DIR

mkdir -p build
cd build
cmake ..

make -j4

mv dramsim3main ../