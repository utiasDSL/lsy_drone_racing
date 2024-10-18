Simulation
==========

The first part of the challenge is to complete the race track in simulation. To that end, we provide a high-fidelity simulator with interfaces that are identical to the real-world setup. We use PyBullet as the underlying simulator to model the physics of the drone and to render the environment.

Difficulty Levels
-----------------

The challenge is divided into different difficulty levels, each specified by a TOML configuration file. These levels progressively increase in complexity:

.. list-table::
   :header-rows: 1
   :widths: 20 10 15 20 15 20

   * - Evaluation Scenario
     - Constraints
     - Rand. Inertial Properties
     - Randomized Obstacles, Gates
     - Rand. Between Episodes
     - Notes
   * - Level 0 (config/level0.toml)
     - Yes
     - No
     - No
     - No
     - Perfect knowledge
   * - Level 1 (config/level1.toml)
     - Yes
     - Yes
     - No
     - No
     - Adaptive
   * - Level 2 (config/level2.toml)
     - Yes
     - Yes
     - Yes
     - No
     - Learning, re-planning
   * - Level 3 (config/level3.toml)
     - Yes
     - Yes
     - Yes
     - Yes
     - Robustness
   * - sim2real
     - Yes
     - Real-life hardware
     - Yes
     - No
     - Sim2real transfer

.. note::
   "Rand. Between Episodes" (governed by argument `reseed_on_reset`) determines whether randomized properties and positions vary or are kept constant (by re-seeding the random number generator on each `env.reset()`) across episodes.

You may use the easier scenarios to develop and debug your controller. However, the final evaluation will be on the hardest scenario (Level 3).

Switching Between Configurations
--------------------------------

You can choose which configuration to use by changing the `--config` command line option. For example, to run the example controller on the hardest scenario, use the following command:

.. code-block:: bash

   python scripts/sim.py --config level3.toml

To use your own controller, you can pass the path to your controller script as the `--controller` argument. For example:

.. code-block:: bash

   python scripts/sim.py --config level3.toml --controller my_controller.py

.. note::
    You can also write your controller file directly into the config files located in the `config` folder. That way, you don't need to specify the controller script when running the simulation, and the controller will be used in the automated challenge evaluation.