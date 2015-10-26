# Position mode constants
int8 NO_CONTROL=0
int8 JOINT_SPACE=1
int8 TASK_SPACE=2
#int8 PD_GAINS=3

int8 LEFT_ARM=0
int8 RIGHT_ARM=1

int32 id  # ID must be echoed back in SampleResult
int8 mode  # Denotes whether data stores joint angles, end-effector positions, etc.
int8 arm # Which arm to execute on
float64[] data
float64[] pd_gains
