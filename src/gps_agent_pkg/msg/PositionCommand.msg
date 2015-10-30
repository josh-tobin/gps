int32 id  # ID must be echoed back in SampleResult
uint32 mode  # denotes whether data stores joint angles, end-effector positions, etc.
uint32 arm # which arm to execute on (actuator type enum)
float64[] data
float64[] pd_gains
