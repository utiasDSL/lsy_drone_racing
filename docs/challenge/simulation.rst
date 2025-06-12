Simulation
==========

The first part of the challenge is to complete the race track in simulation. To that end, we provide a high-fidelity simulator with interfaces that are identical to the real-world setup. We use `crazyflow <https://github.com/utiasDSL/crazyflow/tree/main>`_ as the underlying simulator to model the physics of the drone and to render the environment.

Running the Simulation
----------------------

Once you have installed the ``lsy_drone_racing`` package, you should be able to start a simulation run with

.. code-block:: bash

   python scripts/sim.py

The simulation script has a built-in CLI that allows you to set a few useful arguments. For example, if you want the simulation to render its output and run more than just a single race, you can pass these arguments with

.. code-block:: bash

   python scripts/sim.py -g -n 10

This will render the simulation (``-g`` or ``--gui``) and run 10 races in succession (``-n`` or ``--n_runs``). For a list of all arguments, use ``-h`` or ``--help``.


Difficulty Levels
-----------------

The challenge is divided into different difficulty levels, each specified by a TOML configuration file. These levels progressively increase in complexity:

.. list-table::
   :header-rows: 1
   :widths: 20 15 20 20

   * - Evaluation Scenario
     - Rand. Inertial Properties
     - Randomized Obstacles, Gates
     - Notes
   * - Level 0 (`config/level0.toml <https://github.com/utiasDSL/lsy_drone_racing/blob/main/config/level0.toml>`_)
     - No
     - No
     - Perfect knowledge
   * - Level 1 (`config/level1.toml <https://github.com/utiasDSL/lsy_drone_racing/blob/main/config/level1.toml>`_)
     - Yes
     - No
     - Adaptive
   * - Level 2 (`config/level2.toml <https://github.com/utiasDSL/lsy_drone_racing/blob/main/config/level2.toml>`_)
     - Yes
     - Yes
     - Learning, re-planning
   * - sim2real
     - Real-life hardware
     - Yes
     - Sim2real transfer
   * - Bonus (`config/multi_level3.toml <https://github.com/utiasDSL/lsy_drone_racing/blob/main/config/multi_level3.toml>`_)
     - Yes
     - Yes
     - Multi-agent racing

You may use the easier scenarios to develop and debug your controller. However, the final evaluation will be on the hardest scenario (Level 2) and, more importantly, the sim2real scenario.

Switching Between Configurations
--------------------------------

You can choose which configuration to use by changing the ``--config`` command line option. For example, to run the example controller on the hardest scenario, use the following command:

.. code-block:: bash

   python scripts/sim.py --config level2.toml

To use your own controller, you can pass the path to your controller script as the ``--controller`` argument. For example:

.. code-block:: bash

   python scripts/sim.py --config level2.toml --controller my_controller.py

.. note::
    You can also write your controller file directly into the config files located in the ``config`` folder. That way, you don't need to specify the controller script when running the simulation, and the controller will be used in the automated challenge evaluation.