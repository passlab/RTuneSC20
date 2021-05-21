#!/bin/bash

if [ -z "$1" ]
then
    FILENAME_PREFIX="rtune"
else
    FILENAME_PREFIX=$1
fi

if [ -z "$2" ]
then
    START_SIZE=32
else
    START_SIZE=$2
fi

if [ -z "$3" ]
then
    END_SIZE=512
else
    END_SIZE=$3
fi

if [ -z "$4" ]
then
    STEP=32
else
    STEP=$4
fi

if [ -z "$5" ]
then
    EXECUTABLE_POSTFIX="rtune"
else
    EXECUTABLE_POSTFIX=$5
fi

echo "Max Problem Size, CPU, GPU, Linear Regression" > ${FILENAME_PREFIX}_random.csv

for (( i = ${START_SIZE}; i <= ${END_SIZE}; i += ${STEP} ))
  do 
    echo $i
    ./amr_stencil_${EXECUTABLE_POSTFIX}.out $i 2 >> ${FILENAME_PREFIX}_random.csv
 done
