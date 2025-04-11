Overview
========

The Drone
~~~~~~~~~
We use the Crazyflie nano quadcopter for our challenge. It is a small, low-cost drone that is easy to program and control. You can find more information about the drone at https://www.bitcraze.io/. Crazyflies have the advantage of being fully open-source and having a large community, which makes it easy to find libraries and tools for programming and controlling the drone. Furthermore, their low price reduces the barrier to entry for participants, and the loss of a drone during the competition is less painful.

The Track
~~~~~~~~~
Contrary to other drone racing challenges, we not only include gates in our tracks, but also obstacles designed to be avoided by the drones. The current iteration uses four gates and four obstacles. Gates have to be traversed in the correct order and in the correct direction. Passing a gate in the opposite direction will not count as a successful pass.

Project Goals
~~~~~~~~~~~~~
The goal of the project is to develop a controller that can navigate the drone through the track with the least possible time on a real Crazyflie drone. As previously mentioned, controllers have to be compatible with the predefined interface. Once you have designed a controller and verified its performance and safety in simulation, you can deploy it on the real hardware.

.. warning::
    While the interfaces are the same and we take great care to ensure that the simulation faithfully reproduces the real-world behavior of the drones, running a controller on the real drone is significantly different from running it in simulation. Be aware that unexpected behavior might occur, depending on the actual drone and the used controller, and further extensive tuning and testing may be necessary.
