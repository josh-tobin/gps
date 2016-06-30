#!/bin/bash

python python/gps/parallel_gps.py crl_rnn_ml_avn128 crl_rnn_ml_asn128 crl_rnn_ml_amm128_lessdata
python policy_tester.py crl_rnn_ml_avn128 crl_rnn_ml_avn128
python policy_tester.py crl_rnn_umm12 crl_rnn_ml_avn128
python policy_tester.py crl_rnn_ml_asn128 crl_rnn_ml_asn128
python policy_tester.py crl_rnn_umm12 crl_rnn_ml_asn128
python policy_tester.py crl_rnn_ml_amm128_lessdata crl_rnn_ml_amm128_lessdata
python policy_tester.py crl_rnn_ml_umm12 crl_rnn_ml_amm128_lessdata
(python print_results.py crl_rnn_ml_avn128; python print_results.py crl_rnn_ml_asn128; python print_results.py crl_rnn_ml_amm128_lessdata) | mail -s "Results" joshp.tobin@gmail.com
