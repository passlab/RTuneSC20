#!/bin/bash
for i in {2..36..2}
do 
    ./lulesh2.0-clean -s 20 -t $i
done

for i in {2..36..2}
do
    ./lulesh2.0-clean -s 30 -t $i
done

for i in {2..36..2}
do
    ./lulesh2.0-clean -s 40 -t $i
done

export LD_LIBRARY_PATH=../rtune-install/lib:$LD_LIBRARY_PATH
export RTUNE_CONFIGFILE=./rtune_LULESH_config.txt

./lulesh2.0-rtune -s 20
./lulesh2.0-rtune -s 30
./lulesh2.0-rtune -s 40
