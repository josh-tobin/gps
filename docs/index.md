
Guided Policy Search
===

This codebase implements the code of several research papers involving guided policy search. It includes a complete robot controller and sensor interface
for the PR2 robot via ROS, and an interface for simulated agents in Box2D and Mujoco.
Source code is available on [GitHub](https://github.com/cbfinn/gps).

*****

## What does it do?
This code is the base of several research papers involving guided policy search, including:
* Sergey Levine\*, Chelsea Finn\*, Trevor Darrell, Pieter Abbeel. *End-to-End Training of Deep Visuomotor Policies*. 2015. arxiv 1504.00702. [[pdf](http://arxiv.org/pdf/1504.00702.pdf)]
* Marvin Zhang, Zoe McCarthy, Chelsea Finn, Sergey Levine, Pieter Abbeel. *Learning Deep Neural Network Policies with Continuous Memory States*. ICRA 2016. [[pdf](http://arxiv.org/pdf/1507.01273.pdf)]
* Chelsea Finn, Xin Yu Tan, Yan Duan, Trevor Darrell, Sergey Levine, Pieter Abbeel. *Deep Spatial Autoencoders for Visuomotor Learning*. ICRA 2016.  [[pdf](http://arxiv.org/pdf/1509.06113.pdf)]
* Justin Fu, Sergey Levine, Pieter Abbeel. *One-shot Learning of Manipulation Skills with Online Dynamics Adaptation and Neural Network Priors*. 2016. arxiv 1509.06841.  [[pdf](http://arxiv.org/pdf/1509.06841.pdf)]

Please cite the relevant paper(s) above and the following if the codebase is helpful for your research:
* TODO

To faciliate applying the method to alternative platforms, this codebase also includes complete interfaces to three robot/simulation platforms: Box2D, Mujoco, and ROS.

## Installation

**Dependencies**

The following are required
* Python 2.7, Numpy
* Boost, boost-python
* protobuf

One of the following neural network libraries is required for the full guided policy search algorithm
* Caffe (master branch as of 11/2015 or later, with pycaffe compiled)
* Theano & Lasagne (coming soon)
* CGT & Lasagne (coming soon)

Optional agent interfaces
* Box2D
* ROS
* Mujoco

**Setup**

After installing the necessary dependencies:

1. Clone the repo:

    ```sh
    git clone https://github.com/cbfinn/gps.git
    ```

2. Set up paths:

    ```sh
    export GPS_PATH=/path/to/gps
    export PYTHONPATH=$PYTHONPATH:$GPS_PATH/python
    ```

3. Compile protobuffer:

    ```sh
    cd $GPS_PATH
    ./compile_proto.sh
    ```
**Box2D Setup**

1. TODO

**Mujoco Setup**

1. Install Mujoco

2. Talk to Marvin

3. Set up paths:
    ```sh
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GPS_PATH/lib:$GPS_PATH/build/lib
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

## Box2D example (Box2D required)
TODO
## PR2 example (ROS required)
TODO

## Running a new experiment
1. Make a new directory for your experiment in the experiments/ directory (e.g. `mkdir ./experiments/my_experiment`)

2. Add a hyperparams.py file to your directory. See [pr2_example](https://github.com/cbfinn/gps/blob/master/experiments/pr2_example/hyperparams.py) and [mjc_example](https://github.com/cbfinn/gps/blob/master/experiments/mjc_example/hyperparams.py) for examples.

3. Run the following:
    ```sh
    cd GPS_PATH
    python python/gps/gps_main.py my_experiment
    ```

All of the output logs and data will be routed to your experiment directory.


## Learning with your own robot
The code was written to be modular, to make it easy to hook up your own robot. :)
## Target Setup Docs
TODO
## ROS Controller Docs
TODO
## Reporting bugs and getting help
You can post your queries on this email group list. If you want to contribute, post on this discussion group, then make a pull request on github.
