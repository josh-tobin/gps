#!/bin/bash

python python/gps/parallel_gps.py crl_jar_svn40 crl_jar_svv40
(python policy_tester.py crl_jar_svn40 crl_jar_svn40; python policy_tester.py crl_jar_svv40 crl_jar_svv40) | mail -s "Result of experiment 4" joshp.tobin@gmail.com

