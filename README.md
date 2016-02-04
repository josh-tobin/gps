GPS
======
Guided Policy Search

## Dependencies
The following are required
* Python 2.7, Numpy
* Boost, boost-python
* protobuf

One of the following neural network libraries is required for the full guided policy search algorithm
* Caffe (master branch as of 11/2015 or later, with pycaffe compiled)
* Theano & Lasagne (coming soon)
* CGT & Lasagne (coming soon)

Optional agent interfaces
* ROS
* Mujoco
* Box2D (coming soon)

## Setup
Follow the following steps to get set up

1. Install necessary dependencies above.

2. Clone this repo:

    ```sh
    git clone https://github.com/cbfinn/gps.git
    ```

3. Set up paths:

    ```sh
    export GPS_PATH=/path/to/gps
    export PYTHONPATH=$PYTHONPATH:$GPS_PATH/python
    ```

4. Compile protobuffer:

    ```sh
    cd $GPS_PATH
    ./compile_proto.sh
    ```

**Mujoco Setup**

In addition to the dependencies listed above, OpenSceneGraph is also needed.

1. [Install Mujoco](https://www.roboti.us/) and place the downloaded `mjpro` directory into `$GPS_PATH/src/3rdparty`. Obtain a key, which should be named `mjkey.txt`, and place the key into the `mjpro` directory.

2. Build `$GPS_PATH/src/3rdparty`. Run `cd $GPS_PATH && mkdir build && cd build`, then `cmake ../src/3rdparty`, then `make -j`.

3. Set up paths:
    ```sh
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GPS_PATH/build/lib
    export PYTHONPATH=$PYTHONPATH:$GPS_PATH/build/lib
    ```


**ROS Setup**

1. Install ROS

2. Set up paths:

    ```sh
    export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$GPS_PATH:$GPS_PATH/src/gps_agent_pkg
    ```
3. Compilation:

    ```sh
    cd src/gps_agent_pkg/
    cmake .
    make -j
    ```

## Intended Usage
1. Make a new directory for your experiment in the experiments/ directory (e.g. `mkdir ./experiments/my_experiment`)

2. Add a hyperparams.py file to your directory. See [pr2_example](https://github.com/cbfinn/gps/blob/master/experiments/pr2_example/hyperparams.py) and [mjc_example](https://github.com/cbfinn/gps/blob/master/experiments/mjc_example/hyperparams.py) for examples.

3. Run the following:
    ```sh
    cd GPS_PATH
    python python/gps/gps_main.py my_experiment
    ```

All of the output logs and data will be routed to your experiment directory.
