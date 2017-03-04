#!/bin/bash

for layers in 1 2; do
    for batch_size in 2 8 32; do
        for initial_lr in 0.1 0.01 0.001; do
            for cell_size in 16 64 128; do
                for dropout in 0 0.25; do
                    echo python char_rnn.py --layers $layers --batch_size $batch_size --initial_lr $initial_lr --cell_size $cell_size --dropout $dropout --vocabulary vocabulary.pkl --example_file examples.pb --minutes_to_train 30
                done
            done
        done
    done
done
