#!/bin/bash

DIR1="data"
DIR2="data2"

# look for empty data1/data2
if [ "$(ls -A $DIR1)" ]; then
    echo "$DIR1 is not Empty"
else
    sudo rm -rf data;
    echo "delete data";
fi

if [ "$(ls -A $DIR2)" ]; then
    echo "$DIR2 is not Empty"
else
    sudo rm -rf data2;
    echo "deleted data2";
fi
