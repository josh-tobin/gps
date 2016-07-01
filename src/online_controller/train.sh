#!/bin/bash
#for cc in {1..7}
#do
#    for i in {1..20}
#    do
#        python mjc_interface.py --condition ${cc} -T 1200 -s -v
#    done
#done

#python train_rnn.py net/rnn/net4.pkl 4 > net/rnn/net4.log
python train_rnn.py net/rnn/net5.pkl 5 > net/rnn/net5.log
python train_rnn.py net/rnn/net6.pkl 6 > net/rnn/net6.log
#python train_rnn.py net/rnn/net7.pkl 7 > net/rnn/net7.log
#python train_rnn.py net/rnn/net8.pkl 8 > net/rnn/net8.log
#python train_rnn.py net/rnn/net9.pkl 9 > net/rnn/net9.log

