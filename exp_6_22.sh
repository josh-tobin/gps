#!/bin/bash

python python/gps/parallel_gps.py crl_rnn_ml_nvn8 crl_rnn_ml_nvv8
(python policy_tester.py crl_rnn_ml_nvn8 crl_rnn_ml_nvn8; python policy_tester.py crl_rnn_ml_nvv8 crl_rnn_ml_nvv8) | mail -s "Normally distributed experiments" joshp.tobin@gmail.com

python python/gps/parallel_gps.py crl_rnn_ml_usn16 crl_rnn_ml_uss16 crl_rnn_ml_uvn16 crl_rnn_ml_uvv16
(python policy_tester.py crl_rnn_ml_usn16 crl_rnn_ml_usn16; python policy_tester.py crl_rnn_ml_uss16 crl_rnn_ml_uss16; python policy_tester.py crl_rnn_ml_uvn16 crl_rnn_ml_uvn16; python policy_tester.py crl_rnn_ml_uvv16 crl_rnn_ml_uvv16) | mail -s "16 condition experiments" joshp.tobin@gmail.com

python python/gps/parallel_gps.py crl_rnn_ml_usn32 crl_rnn_ml_uss32
(python policy_tester.py crl_rnn_ml_usn32 crl_rnn_ml_usn32; python policy_tester.py crl_rnn_ml_uss32 crl_rnn_ml_uss32) | mail -s "32 condition experiments" joshp.tobin@gmail.com
