# Position mode constants
int8 JOINT_ANGLE=1
int8 END_EFFECTOR=2
int8 PD_GAINS=3

# Arm constants
int8 LEFT_ARM=1
int8 RIGHT_ARM=2

Header header
int8 mode  # Denotes whether data stores joint angles, end-effector positions, etc.
int8 arm # Which arm to execute on
float64[] data
