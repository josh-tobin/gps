#!/bin/bash
for cc in {0..9}
do
    for i in {1..4}
    do
        python mjc_driver.py --condition ${cc} -T 800 -s online_data.mat -n --config config_basic config_nn --silence_logger
        python mjc_driver.py --condition ${cc} -T 800 -s online_data.mat -n --config config_basic2 config_nn --silence_logger
        python mjc_driver.py --condition ${cc} -T 800 -s online_data.mat -n --config config_basic3 config_nn --silence_logger
        python mjc_driver.py --condition ${cc} -T 800 -s online_data.mat -n --config config_basic4 config_nn --silence_logger
        python mjc_driver.py --condition ${cc} -T 800 -s online_data.mat -n --config config_basic5 config_nn --silence_logger
        echo ${i}
    done
    echo "Finished condition "${cc}
done
