# /bin/bash

python python/gps/parallel_gps.py crl_rnn_1mass_ramp crl_rnn_1mass_ramp_highp crl_rnn_1mass_highp
(python print_results.py crl_rnn_1mass_ramp; python print_results.py crl_rnn_1mass_ramp_highp; python print_results.py crl_rnn_1mass_highp) | mail -s "High torque penalty experiments" joshp.tobin@gmail.com

python python/gps/parallel_gps.py crl_brick_1mass crl_brick_8mass crl_brick_32mass
(python print_results.py crl_brick_1mass; python print_results.py crl_brick_8mass; python print_results.py crl_brick_32mass) | mail -s "Brick experiments" joshp.tobin@gmail.com
