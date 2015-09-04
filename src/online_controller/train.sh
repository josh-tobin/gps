#!/bin/bash
#for cc in {1..7}
#do
#    for i in {1..200}
#    do
#        python mjc_interface.py --condition ${cc} -T 1200 -s
#    done
#done

python train_rnn.py net/mjc_gru_drop.pkl 1
python train_rnn.py net/mjc_simplegate.pkl 2
python train_rnn.py net/mjc_gru_small.pkl 3

