# This message is published to the C++ controller to start
# a trial
time stamp

ControllerParams controller

# Trial information
int8 T  # Trajectory length
float64 frequency  # Controller frequency
int8[] state_datatypes  # Which data types to include in state
int8[] obs_datatypes # Which data types to include in observation