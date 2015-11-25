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
    export PYTHONPATH=$PYTHONPATH:GPS_PATH/python/gps
    ```

4. Compile protobuffer:

    ```sh
    cd GPS_PATH
    ./compile_proto.sh
    ```

**Mujoco Setup**

1. Install Mujoco

2. Talk to Marvin

3. Set up paths:
    ```sh
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:GPS_PATH/lib:GPS_PATH/build/lib
    ```


**ROS Setup**

1. Install ROS

2. Set up paths:

    ```sh
    export ROS_PACKAGE_PATH:$ROS_PACKAGE_PATH:GPS_PATH:GPS_PATH/src/gps_agent_pkg
    ```
3. Compilation:

    ```sh
    cd src/gps_agent_pkg/
    cmake .
    make -j
    ```

## Intended Usage
