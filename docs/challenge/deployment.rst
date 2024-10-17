Deployment
==========

The idea for the deployment is to have an environment that exactly matches the interfaces and observation space of the simulation. Since the action space for the controller is identical, controllers can be directly deployed on the real drone without any modifications.

.. warning::
    Please be aware that running a controller on the real drone may still exhibit significant differences compared to the simulation due to the sim2real gap.

Motion Tracking
~~~~~~~~~~~~~~~
We use a Vicon motion tracking system to track the motion of the drone. The Vicon system consists of several cameras that are placed around the track, and a base station that calculates object poses by triangulation. Gates, obstacles and the drone are all equipped with reflective markers, which can be tracked by the cameras. We use the Vicon bridge from ETH's ASL to send the poses into ROS.

Deploying Your Controller
~~~~~~~~~~~~~~~~~~~~~~~~~
To deploy your controller on the real drone, use the deployment script in the ``lsy_drone_racing/scripts`` folder. Before running the script, make sure to set the correct drone parameters in the Crazyswarm package. Place the drone on its start position, power it on, and launch the Crazyswarm ROS node with 

.. code-block:: bash

    roslaunch crazyswarm hover_swarm.launch

.. note::
    Make sure the drone has enough battery to complete the track. If a red LED is constantly turned on, the drone is low on battery. A blinking red LED indicates that the battery is sufficiently charged.

.. note::
    You should restart the drone after every flight to reset the internal state estimators and its command mode.

.. warning::
    Only turn on the drone once you placed it on the start position. The internal sensors of the drone are reset on power-on, and turning it on while you are still moving it around may cause internal estimators to drift.

This should open up an RViz window with the drone frame and the world frame. You can now launch your controller by running

.. code-block:: bash

    python scripts/deploy_controller.py --controller <controller_name>

The deployment script will first check if the real track poses and the drone starting pose is within acceptable bounds of the configured track. If not, the script will print an error message and terminate. If the poses are correct, the drone will take off, fly through the track, print out the final lap time, and land automatically.
