#!/bin/bash

DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd );
cd ${DIR}

mkdir build
mkdir install

cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ../
# make
make install