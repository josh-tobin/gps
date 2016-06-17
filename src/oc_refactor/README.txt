
0. If you are using the MuJoCo compiled on this branch,
    place mjkey.txt into your working directory (there should be a message).

    This path was hardcoded in the original code in
    3rdparty/mjcpy2/mjcpy2.cpp line 92, so you can change this
    behavior if you want

[DONE] 1. Collect data for training dynamics

    $ python mjc_driver.py --offline

    You can adjust the hyperparams in 
        1) mjc_driver.py:setup_agent
        2) mjc_driver.py:setup_algorithm
    And adjust the iterations/number of samples taken in
    mjc_driver.py:run_offline

    This function will create a file under data/offline_dynamics_data.mat

[TODO: Test] 2. Train a dynamics neural network
    
    $ python train_nn.py
    
[TODO: Test] 3. Run online controller
    
    $ python mjc_driver.py -T <time_steps>

    Configuration:
        TODO
