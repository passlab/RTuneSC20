#!/bin/bash

export LD_LIBRARY_PATH=../rtune-install/lib:$LD_LIBRARY_PATH
export RTUNE_CONFIGFILE=./rtune_LULESH_config.txt

./lulesh2.0-rtune -s 20
./lulesh2.0-rtune -s 30
./lulesh2.0-rtune -s 40
