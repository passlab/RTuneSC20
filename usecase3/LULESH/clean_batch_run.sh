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
