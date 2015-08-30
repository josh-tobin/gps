#!/bin/bash
for cc in {1..7}
do
    for i in {1..200}
    do
        python mjc_interface.py --condition ${cc} -T 1200 -s
    done
done


