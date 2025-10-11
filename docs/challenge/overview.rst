Overview
========

The Drone
---------
We use the Crazyflie nano quadcopter for our challenge. It is a small, low-cost drone that is easy to program and control. You can find more information about the drone at https://www.bitcraze.io/. Crazyflies have the advantage of being fully open-source and having a large community, which makes it easy to find libraries and tools for programming and controlling the drone. Furthermore, their low price reduces the barrier to entry for participants, and the loss of a drone during the competition is less painful.

The Track
---------
Contrary to other drone racing challenges, we not only include gates in our tracks, but also obstacles designed to be avoided by the drones. The current iteration uses four gates and four obstacles. Gates have to be traversed in the correct order and in the correct direction. Passing a gate in the opposite direction will not count as a successful pass.

Project Goals
-------------
The goal of the project is to develop a controller that can navigate the drone through the track with the least possible time on a real Crazyflie drone. As previously mentioned, controllers have to be compatible with the predefined interface. Once you have designed a controller and verified its performance and safety in simulation, you can deploy it on the real hardware.

.. warning::
    While the interfaces are the same and we take great care to ensure that the simulation faithfully reproduces the real-world behavior of the drones, running a controller on the real drone is significantly different from running it in simulation. Be aware that unexpected behavior might occur, depending on the actual drone and the used controller, and further extensive tuning and testing may be necessary.

Difficulty Levels
-----------------

The challenge is divided into different difficulty levels, each specified by a TOML configuration file. These levels progressively increase in complexity:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 20

   * - Evaluation Scenario
     - Rand. Inertial Properties
     - Randomized Obstacles, Gates
     - Randomized Tracks
     - Notes
   * - Level 0 (`config/level0.toml <https://github.com/utiasDSL/lsy_drone_racing/blob/main/config/level0.toml>`_)
     - No
     - No
     - No
     - Perfect knowledge
   * - Level 1 (`config/level1.toml <https://github.com/utiasDSL/lsy_drone_racing/blob/main/config/level1.toml>`_)
     - Yes
     - No
     - No
     - Adaptive
   * - Level 2 (`config/level2.toml <https://github.com/utiasDSL/lsy_drone_racing/blob/main/config/level2.toml>`_)
     - Yes
     - Yes
     - No
     - Learning, re-planning
   * - Level 3 (`config/level3.toml <https://github.com/utiasDSL/lsy_drone_racing/blob/main/config/level3.toml>`_)
     - Yes
     - Yes
     - Yes
     - Learning, re-planning
   * - sim2real
     - Real-life hardware
     - Yes
     - Yes
     - Yes
     - Sim2real transfer
..    * - Bonus (`config/multi_level3.toml <https://github.com/utiasDSL/lsy_drone_racing/blob/main/config/multi_level3.toml>`_)
..      - Yes
..      - Yes
..      - Multi-agent racing

You may use the easier scenarios to develop and debug your controller. However, the final evaluation will be on the hardest scenario (Level 3) and, more importantly, the sim2real scenario.