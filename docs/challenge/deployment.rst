Deployment
==========

The idea for the deployment is to have an environment that exactly matches the interfaces and observation space of the simulation. Since the action space for the controller is identical, controllers can be directly deployed on the real drone without any modifications.

.. warning::
    Please be aware that running a controller on the real drone may still exhibit significant differences compared to the simulation due to the sim2real gap.

Motion Tracking
~~~~~~~~~~~~~~~
We use a Vicon motion tracking system to track the motion of the drone. The Vicon system consists of several cameras that are placed around the track, and a base station that calculates object poses by triangulation. Gates, obstacles and the drone are all equipped with reflective markers, which can be tracked by the cameras. Since we'd need to resort to numerical differentiation to get velocity information, we're running state estimators that filter the noisy Vicon measurements and provide smoother estimates of the drone's state.

Deploying Your Controller
~~~~~~~~~~~~~~~~~~~~~~~~~
To deploy your controller on the real drone, use the deployment script in the ``lsy_drone_racing/scripts`` folder. Place the drone on its start position, power it on, and launch the estimators.

.. note::
    Make sure the drone has enough battery to complete the track. If a red LED is constantly turned on, the drone is low on battery. A blinking red LED indicates that the battery is sufficiently charged.

The estimators require two packages to run. First, you need to launch the ``motion_capture_tracking`` package to publish the Vicon poses to ROS2.

.. code-block:: bash

    mamba activate race
    ros2 launch motion_capture_tracking launch.py

Next, you also need to launch the estimator for the drone. Note that you need to modify the ``estimators.toml`` file in the ``estimators`` repository to include the correct drone id.

This should open up an RViz window with the drone frame and the world frame.

.. warning::
    If you cannot see the drone in RVIZ, it is likely that Vicon is not turned on, or the drone is not selected for tracking in the Vicon system.

Once everything is running, you can launch your controller by executing

.. code-block:: bash

    python scripts/deploy_controller.py --controller <controller_name>

The deployment script will first check if the real track poses and the drone starting pose is within acceptable bounds of the configured track. If not, the script will print an error message and terminate. If the poses are correct, the drone will take off, fly through the track, print out the final lap time, and land automatically.
