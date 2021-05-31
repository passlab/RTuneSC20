#!/bin/bash

export HARMONY_HOME=${HOME}/harmony
for i in 20 30 40
do
./lulesh2.0-ah -s $i
done
