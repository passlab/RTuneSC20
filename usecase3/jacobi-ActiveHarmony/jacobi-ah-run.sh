#!/bin/bash

export HARMONY_HOME=${HOME}/harmony
for i in 64 128 256 512
do
./jacobi-ah $i
done
