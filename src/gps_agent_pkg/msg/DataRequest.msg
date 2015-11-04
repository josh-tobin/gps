# Publish to C++ controller to request the current value of a single data_type
# (ex. end effectors, joint angles, etc.)
int32 id  # ID must be echoed back in SampleResult
int8 data_type  # sample type enum, defined in gps_pb2
int8 arm
