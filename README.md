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

2. Clone this repo.
```sh
git clone https://github.com/cbfinn/gps.git
```

3. Set up paths
```
export PYTHONPATH=$PYTHONPATH:/path/to/gps:/path/to/gps/python/gps:/path/to/gps/lib:/path/to/gps/python/gps/algorithm/policy_opt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/gps/lib:/path/to/gps/build/lib
```

4. Compile protobuffer
```sh
cd /path/to/gps
./compile_proto.sh
```

**Mujoco Setup**

1. Install Mujoco

2. Set up paths

**ROS Setup**

3. Install ROS

4. Set up paths
```
export ROS_PACKAGE_PATH:$ROS_PACKAGE_PATH:/path/to/gps:/path/to/gps/src/gps_agent_pkg
```

3. Compilation
```
cd src/gps_agent_pkg/
cmake .
make -j
```
