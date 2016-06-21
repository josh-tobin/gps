
0. If you are using the MuJoCo compiled on this branch,
    place mjkey.txt into your working directory (there should be a message).

    This path was hardcoded in the original code in
    3rdparty/mjcpy2/mjcpy2.cpp line 92, so you can change this
    behavior if you want

[Optional] 1. Collect data for training dynamics

    $ python mjc_driver.py --offline [--noverbose]

    The option --noverbose Turns off plotting for faster data collection once everything is verified to work.

    You can adjust the hyperparams in
        1) mjc_driver.py:setup_agent
        2) mjc_driver.py:setup_algorithm
    And adjust the iterations/number of samples taken in
    mjc_driver.py:run_offline

    This function will create a file under data/offline_dynamics_data.mat


    You can also modify/run collect_data.sh to run the online controller to
    collect more data.

[Optional] 2. Train a dynamics neural network

    $ python train_nn.py --netid <network_name>

    You can adjust the architecture inside the function train_nn.py:train_nn
    Layers are defined in dynamics_nn.py

    network_name defaults to 'contextual' (see the build_network function for
    its definition)

    Dataflow between layers is defined using names
        Ex. the ReLULayer has the signature
        ReLULayer(<input_layer_name>, <output_layer_name>)
        So you would do this to chain a relu with an inner product layer:

        ff1 = FFIPLayer('data', 'ip1')
        relu1 = ReLULayer('ip1', 'relu1')
        ff2 = FFIPLayer('relu1', 'lbl')

        'data' is always defined as the input.

    For training, you can adjust the line
        net.init_functions(output_blob='acc', weight_decay=1e-4,
            train_algo='rmsprop')

        train_algo can either be 'rmsprop' or 'sgd'

        The lr and lr_schedule are defined below that line.


3. Run online controller

    $ python mjc_driver.py -T <time_steps> --condition <condition> --config
    <config list>

    The --config option requests a list of config files.
        Ex. To use the nn priot
        --config config_basic config_nn
        or to use the GMM prior
        --config config_basic config_gmm 

        The config files are applied in order - the most recent one overrides
        the previous ones. These config files are defined as python scripts
        (ex. config_basic.py, config_nn.py)



