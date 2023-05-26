#!/bin/bash

workspace=/content/drive/MyDrive
program=$workspace/src/__main__.py
export PYTHONPATH=$workspace/src:$PYTHONPATH

python $program --surface_all --NNmodel 1 --dataset 0 --dropout 0
python $program --surface_avg --NNmodel 1 --dataset 0 --dropout 1
python $program --surface_avg --NNmodel 1 --dataset 1 --dropout 0
python $program --surface_avg --adam --NNmodel 1 --dataset 1 --dropout 0

for i in {1..8}; do
    echo Running iteration $i
    if [ $i -le 4 ]; then
        adam_arg=""
    else
        adam_arg="--adam"
    fi
    dataset_arg=$((($i - 1) / 2 % 2))
    dropout_arg=$((($i - 1) % 2))
    python $program --train --step $adam_arg --NNmodel 1 --dataset $dataset_arg --dropout $dropout_arg
done

for i in {1..8}; do
    echo Running iteration $i
    if [ $i -le 4 ]; then
        adam_arg=""
    else
        adam_arg="--adam"
    fi
    dataset_arg=$((($i - 1) / 2 % 2))
    dropout_arg=$((($i - 1) % 2))
    python $program --train --step $adam_arg --NNmodel 3 --dataset $dataset_arg --dropout $dropout_arg
done

for i in {1..8}; do
    echo Running iteration $i
    if [ $i -le 4 ]; then
        adam_arg=""
    else
        adam_arg="--adam"
    fi
    dataset_arg=$((($i - 1) / 2 % 2))
    dropout_arg=$((($i - 1) % 2))
    python $program --train --step $adam_arg --NNmodel 4 --dataset $dataset_arg --dropout $dropout_arg
done

for i in {1..8}; do
    echo Running iteration $i
    if [ $i -le 4 ]; then
        adam_arg=""
    else
        adam_arg="--adam"
    fi
    dataset_arg=$((($i - 1) / 2 % 2))
    dropout_arg=$((($i - 1) % 2))
    python $program --train --step $adam_arg --NNmodel 2 --dataset $dataset_arg --dropout $dropout_arg
done

