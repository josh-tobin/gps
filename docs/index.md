
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

#### Dependencies

The following are required
* [python 2.7](https://www.python.org/download/releases/2.7/), [numpy](http://www.numpy.org), [matplotlib](http://matplotlib.org)
* [boost](http://www.boost.org/), including boost-python
* [protobuf](https://developers.google.com/protocol-buffers/)

One or more of the following agent interfaces is required. Set up instructions for each are below.
* [Box2D](https://github.com/pybox2d/pybox2d)
* [ROS](http://ros.org)
* [Mujoco](https://www.roboti.us/)

One of the following neural network libraries is required for the full guided policy search algorithm
* [Caffe](http://caffe.berkeleyvision.org/) (master branch as of 11/2015, with pycaffe compiled, python layer enabled, PYTHONPATH configured)
* [TensorFlow](http://tensorflow.org) (coming soon)

#### Setup

Follow the following steps to get set up:

1. Install necessary dependencies above.

2. Clone the repo:

    ```sh
    git clone https://github.com/cbfinn/gps.git
    ```

4. Compile protobuffer:

    ```sh
    cd gps
    ./compile_proto.sh
    ```

**Box2D Setup** (optional)

Here are the instructions for setting up [Pybox2D](https://github.com/pybox2d/pybox2d).

1. Install Swig and Pygame:

    ```sh
    sudo apt-get install build-essential python-dev swig python-pygame subversion
    ```
2. Check out the Pybox2d code via SVN

    ```sh
    svn checkout http://pybox2d.googlecode.com/svn/trunk/ pybox2d
    ```

3. Build and install the library:

    ```sh
    python setup.py build
    sudo python setup.py install
    ```

**Mujoco Setup** (optional)

In addition to the dependencies listed above, OpenSceneGraph is also needed.

1. [Install Mujoco](https://www.roboti.us/) and place the downloaded `mjpro` directory into `gps/src/3rdparty`. Obtain a key, which should be named `mjkey.txt`, and place the key into the `mjpro` directory.

2. Build `gps/src/3rdparty` by running:
    ```sh
    cd gps && mkdir build && cd build
    cmake ../src/3rdparty
    make -j
    ```

3. Set up paths by adding the following to your `~/.bashrc` file:
    ```sh
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:gps/build/lib
    export PYTHONPATH=$PYTHONPATH:gps/build/lib
    ```
    Don't forget to run `source ~/.bashrc` afterward.

**ROS Setup** (optional)

1. Install ROS.

2. Set up paths by adding the following to your `~/.bashrc` file:

    ```sh
    export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/path/to/gps:/path/to/gps/src/gps_agent_pkg
    ```
    Don't forget to run `source ~/.bashrc` afterward.
3. Compilation:

    ```sh
    cd src/gps_agent_pkg/
    cmake .
    make -j
    ```

**ROS Setup with Caffe** (optional)

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

There are two examples of running trajectory optimizaiton using a simple 2D agent in Box2D. Before proceeding, be sure to [set up Box2D](#setup).

The first is a point mass learning to move to goal position. To try it out, run the following from the base directory of the repo:
```
python python/gps/gps_main.py box2d_pointmass_example
```

The second example is a 2-link arm learning to move to goal state. To try it out, run this:
```
python python/gps/gps_main.py box2d_arm_example
```

Both examples start with a random controller and learn though experience how to reach the goal! All settings for these examples are located in `experiments/box2d_[name]_example/hyperparams.py`,
which can be modified to input different target positions, bring up a GUI that displays learning progress, and change various hyperparameters of the algorihtm.

#### Mujoco example

To run the mujoco example, be sure to first [set up Mujoco](#setup).

The first example is using trajectory optimizing for peg insertion. To try it, run the following:
```
python python/gps/gps_main.py mjc_example
```
Here the robot starts with a random initial controller and learns to insert the peg into the hole.

Now let's learn to generalize to different positions of the hole. For this, run the guided policy search algorithm:
```
python python/gps/gps_main.py mjc_badmm_example
```

The robot learns a neural network policy for inserting the peg under varying initial conditions.

To tinker with the hyperparameters and input, take a look at `experiments/mjc_badmm_example/hyperparams.py`.

#### PR2 example

TODO - Zoe fill this in, including code for launching pr2_real/pr2_sim, code for running gps_main, note on different versions of ROS, note on how to set new targets on the pr2 via targetsetup, with a link to the [gui documentation](gui.html).

#### Running a new experiment
1. Make a new directory for your experiment in the experiments/ directory (e.g. `mkdir ./experiments/my_experiment`)

2. Add a hyperparams.py file to your directory. See [pr2_example](https://github.com/cbfinn/gps/blob/master/experiments/pr2_example/hyperparams.py) and [mjc_example](https://github.com/cbfinn/gps/blob/master/experiments/mjc_example/hyperparams.py) for examples.

3. Run the following:
    ```sh
    cd gps
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
You can post questions on [gps-help](https://groups.google.com/forum/#!forum/gps-help). If you want to contribute,
please post on [gps-dev](https://groups.google.com/forum/#!forum/gps-dev). When your contribution is ready, make a pull request on [GitHub](https://github.com/cbfinn/gps).
