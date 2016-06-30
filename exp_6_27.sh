#!/bin/bash
python python/gps/parallel_gps.py crl_rnn_damping_01 crl_rnn_damping_10
python python/gps/parallel_gps.py crl_rnn_damping_0110 crl_rnn_friction_01
python python/gps/parallel_gps.py crl_rnn_friction_1 crl_rnn_friction_10
python python/gps/parallel_gps.py crl_box_rnn_32mass_widerange crl_box_rnn_8mass
