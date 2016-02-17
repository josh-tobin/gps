
Guided Policy Search
===

This code is a reimplementation of the guided policy search algorithm and LQG-based trajectory optimization, meant to help others understand, reuse, and build upon existing work.
It includes a complete robot controller and sensor interface for the PR2 robot via ROS, and an interface for simulated agents in Box2D and Mujoco.
Source code is available on [GitHub](https://github.com/cbfinn/gps).

While the core functionality is fully implemented and tested, the code base is **a work in progress**. See the [FAQ](faq.html) for information on planned future additions to the code.


*****

## Relevant work

Relevant papers which have used guided policy search include:
* Sergey Levine\*, Chelsea Finn\*, Trevor Darrell, Pieter Abbeel. *End-to-End Training of Deep Visuomotor Policies*. 2015. arxiv 1504.00702. [[pdf](http://arxiv.org/pdf/1504.00702.pdf)]
* Marvin Zhang, Zoe McCarthy, Chelsea Finn, Sergey Levine, Pieter Abbeel. *Learning Deep Neural Network Policies with Continuous Memory States*. ICRA 2016. [[pdf](http://arxiv.org/pdf/1507.01273.pdf)]
* Chelsea Finn, Xin Yu Tan, Yan Duan, Trevor Darrell, Sergey Levine, Pieter Abbeel. *Deep Spatial Autoencoders for Visuomotor Learning*. ICRA 2016.  [[pdf](http://arxiv.org/pdf/1509.06113.pdf)]
* Justin Fu, Sergey Levine, Pieter Abbeel. *One-shot Learning of Manipulation Skills with Online Dynamics Adaptation and Neural Network Priors*. 2016. arxiv 1509.06841.  [[pdf](http://arxiv.org/pdf/1509.06841.pdf)]
* Sergey Levine, Nolan Wagener, Pieter Abbeel. *Learning Contact-Rich Manipulation Skills with Guided Policy Search*. ICRA 2015. [[pdf](http://rll.berkeley.edu/icra2015gps/robotgps.pdf)]
* Sergey Levine, Pieter Abbeel. *Learning Neural Network Policies with Guided Policy Search under Unknown Dynamics*. NIPS 2014. [[pdf](http://www.eecs.berkeley.edu/~svlevine/papers/mfcgps.pdf)]

If the codebase is helpful for your research, please cite any relevant paper(s) above and the following:
* Chelsea Finn, Marvin Zhang, Justin Fu, Xin Yu Tan, Zoe McCarthy, Emily Scharff, Sergey Levine. Guided Policy Search Code Implementation. 2016. Software available from rll.berkeley.edu/gps.

For bibtex, see [this page](bibtex.html).

## Installation

**Dependencies**

The following are required
* Python 2.7, Numpy
* Boost, boost-python
* protobuf

One of the following neural network libraries is required for the full guided policy search algorithm
* Caffe (master branch as of 11/2015, with pycaffe compiled, python layer enabled, PYTHONPATH configured)
* TensorFlow (coming soon)

One or more of the following agent interfaces
* Box2D
* ROS
* Mujoco

#### Setup

Follow the following steps to get set up:

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

**Box2D Setup**

1. TODO

**Mujoco Setup**

In addition to the dependencies listed above, OpenSceneGraph is also needed.

1. [Install Mujoco](https://www.roboti.us/) and place the downloaded `mjpro` directory into `$GPS_PATH/src/3rdparty`. Obtain a key, which should be named `mjkey.txt`, and place the key into the `mjpro` directory.

2. Build `$GPS_PATH/src/3rdparty`. Run:
    ```sh
    cd $GPS_PATH && mkdir build && cd build
    cmake ../src/3rdparty
    make -j
    ```

3. Set up paths:
    ```sh
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GPS_PATH/build/lib
    export PYTHONPATH=$PYTHONPATH:$GPS_PATH/build/lib
    ```

**ROS Setup**

1. Install ROS.

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

**ROS Setup with Caffe**

This is required if you intend to run neural network policies with the ROS agent.

0. Run step 1 and 2 of the above section.

1. Checkout and build caffe, including running `make -j && make distribute` within caffe.

2. Compilation:

    ```sh
    cd src/gps_agent_pkg/
    cmake . -DUSE_CAFFE=1 -DCAFFE_INCLUDE_PATH=/path/to/caffe/distribute/include -DCAFFE_LIBRARY_PATH=/path/to/caffe/build/lib
    make -j
    ```

    To compile with GPU, also include the option `-DUSE_CAFFE_GPU=1`.


## Examples

#### Box2D example

Start by following the above setup for the code repo and box2d.

Then, TODO

#### Mujoco example

Start by following the above setup for the code repo and mujoco.

#### PR2 example

Start by following the above setup for the code repo and ROS.
TODO

#### Running a new experiment
1. Make a new directory for your experiment in the experiments/ directory (e.g. `mkdir ./experiments/my_experiment`)

2. Add a hyperparams.py file to your directory. See [pr2_example](https://github.com/cbfinn/gps/blob/master/experiments/pr2_example/hyperparams.py) and [mjc_example](https://github.com/cbfinn/gps/blob/master/experiments/mjc_example/hyperparams.py) for examples.

3. Run the following:
    ```sh
    cd GPS_PATH
    python python/gps/gps_main.py my_experiment
    ```

All of the output logs and data will be routed to your experiment directory.

## Documentation

In addition to the inline docstrings and comments, see the following pages for additional documentation:

* [GUI for visualization and target setup](gui.html)
* [Configuration and Hyperparameters](hyperparams.html)
* [FAQ](faq.html)

## Learning with your own robot
The code was written to be modular, to make it easy to hook up your own robot. To do so, either use one of the existing agent interfaces (e.g. AgentROS), or write your own.

## Reporting bugs and getting help
You can post your queries on this email group list. If you want to contribute, post on this discussion group, then make a pull request on github.
