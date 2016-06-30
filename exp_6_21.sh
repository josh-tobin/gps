#!/bin/bash
#python python/gps/parallel_gps.py crl_ff_highpenalty crl_jar_ff
#(python policy_tester.py crl_ff_highpenalty crl_ff_highpenalty; python policy_tester.py crl_jar_ff crl_jar_ff) | mail -s "Result of experiment 1" joshp.tobin@gmail.com

python python/gps/parallel_gps.py crl_sll40_smallnet crl_sss40_smallnet 
(python policy_tester.py crl_sll40_smallnet crl_sll40_smallnet; python policy_tester.py crl_sss40_smallnet crl_sss40_smallnet) | mail -s "Result of experiment 2" joshp.tobin@gmail.com

python python/gps/parallel_gps.py crl_svv40 crl_rnn_1mass
(python policy_tester.py crl_svv40 crl_svv40; python policy_tester.py crl_rnn_1mass crl_rnn_1mass) | mail -s "Result of experiment 3" joshp.tobin@gmail.com


python python/gps/parallel_gps.py crl_jar_svn40 crl_jar_svv40
(python policy_tester.py crl_jar_svn40 crl_jar_svn40; python policy_tester.py crl_jar_svv40 crl_jar_svv40) | mail -s "Result of experiment 4" joshp.tobin@gmail.com

