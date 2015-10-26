# Publish to C++ controller to request the current value of a single data_type
# (ex. end effectors, joint angles, etc.)
time stamp
int8 data_type  # Placeholder for enum. These should be defined by src/sample_data/sample_types.h
int8 arm # TODO - this int is defined in RelaxCommand.msg. We should define it somewhere more universal.
