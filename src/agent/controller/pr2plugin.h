/*
This is the PR2-specific version of the robot plugin.
*/
#pragma once

class Pr2Plugin
{
private:
    /* any PR2-specific variables go here */
public:
    // Constructor.
    Pr2Plugin();
    // Destructor.
    virtual ~Pr2Plugin();
    /* the pr2-specific update function should go here;
       this function should do the following:
       - perform whatever housekeeping is needed to note the current time.
       - update all sensors (this might be a no-op for vision, but for
         joint angle "sensors," they need to know the current robot state).
       - update the appropriate controller (position or trial) depending on
         what we're currently doing
       - if the controller wants to send something via a publisher, publish
         that at the end -- it will typically be a completion message that
         includes the recorded robot state for the controller's execution.
     */
};
