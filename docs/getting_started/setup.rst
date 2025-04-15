Installation and Setup
======================

This guide will walk you through the process of setting up the LSY Autonomous Drone Racing project on your system.

Prerequisites
-------------

Before you begin, ensure you have the following:

- Git installed on your system
- A GitHub account
- Either `conda <https://conda.io/projects/conda/en/latest/index.html>`_ or `mamba <https://mamba.readthedocs.io/en/latest/>`_ or one of their flavors installed. We recommend `micromamba <https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>`_.
- Optional: `Docker <https://docs.docker.com/>`_ installed on your system

.. note::
    You can also use `venv` or other dependency management tools, but we won't cover them here.

Required Repositories
---------------------

The LSY Autonomous Drone Racing project requires you to fork the drone racing repository:

`lsy_drone_racing <https://github.com/utiasDSL/lsy_drone_racing>`_ (main branch)

This repository contains the drone simulation, environments, and scripts to simulate and deploy the drones in the racing challenge.

Forking the Repository
----------------------

Start by forking the `lsy_drone_racing <https://github.com/utiasDSL/lsy_drone_racing>`_ repository for your own group. This serves two purposes:

1. You'll have your own repository with git version control.
2. It sets you up for participating in the online competition and automated testing.

If you're new to GitHub, refer to the `GitHub documentation on forking <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_.

Installation Steps
------------------

Follow these steps to set up the project:

#. Clone your forked repository:

   .. code-block:: bash

      mkdir -p ~/repos && cd ~/repos
      git clone https://github.com/<YOUR-USERNAME>/lsy_drone_racing.git

#. Create and activate a new mamba environment:

   You need a working installation of ROS2 Jazzy and Python 3.11 in order to deploy your controller on the real drone. 

   We recommend `RoboStack <https://robostack.github.io>`_ and `micromamba <https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>`_ for this. Robostack lets you install ROS versions independent of you OS. It builds on top of conda/mamba to do this.

   Please follow the `Robostack Getting Started <https://robostack.github.io/GettingStarted.html>`_ in order to create a ROS2 Jazzy Environment on Python 3.11 using micromamba or another package manager of your choice.

   If you just want to use the simulation for local development, you can also install a Python environment without ROS2:

      .. code-block:: bash

         mamba create -n race python=3.11
         mamba activate race

#. Install the lsy_drone_racing package:

   .. code-block:: bash

      cd ~/repos/lsy_drone_racing
      pip install --upgrade pip
      pip install -e .

   If you want to use GPU acceleration, you can install the optional dependencies:

   .. code-block:: bash
      pip install -e .[gpu]

Testing the Installation
------------------------

To verify that the installation was successful:

.. code-block:: bash

   cd ~/repos/lsy_drone_racing
   python scripts/sim.py

If everything is installed correctly, this will open the simulator and simulate a drone flying through four gates.

Extended Dependencies
---------------------

To install extended dependencies for testing:

.. code-block:: bash

   mamba activate race
   cd ~/repos/lsy_drone_racing
   pip install -e .[test]

You can then run the tests to ensure everything is working:

.. code-block:: bash

   cd ~/repos/lsy_drone_racing
   pytest tests

Using Docker
------------

Alternatively, you can run the simulation using Docker, although currently without GUI support:

1. Install Docker with docker compose on your system.
2. Build and run the Docker container:

   .. code-block:: bash

      docker compose --profile sim build
      docker compose --profile sim up

   After building, running the container should produce output similar to:

   .. code-block:: bash

      sim-1  | INFO:__main__:Flight time (s): 8.466666666666667
      sim-1  | Reason for termination: Task completed
      sim-1  | Gates passed: 4
      sim-1  | 
      sim-1  | 8.466666666666667

Docker compose is set up to always reflect the latest changes to the repository without the need to rebuild the image. This does not apply if you have made changes to the dependencies, which requires a rebuild.

.. note::
    We currently do not support running the simulator in GUI mode with Docker, so we recommend using the native installation for easier development.

Troubleshooting
---------------

GLIBCXX Error
^^^^^^^^^^^^^

If you encounter errors related to `LIBGL` and `GLIBCXX_3.4.30` when running the simulation, try the following steps:

#. Run the simulation in verbose mode:

   .. code-block:: bash

      LIBGL_DEBUG=verbose python scripts/sim.py

#. Check if your system has the required library:

   .. code-block:: bash

      strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.30

   Or check in your mamba environment:

   .. code-block:: bash

      strings /<PATH-TP-YOUR-MAMBA>/envs/<ENV-NAME>/lib/libstdc++.so.6 | grep GLIBCXX_3.4.30

#. If the library is missing, install it:

   .. code-block:: bash

      mamba install -c conda-forge gcc=12.1.0

#. If the error persists, update your `LD_LIBRARY_PATH` to include your mamba environment's lib folder.

libNatNet Error (deployment only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If libNatNet is missing during compilation or when launching hover_swarm.launch:

1. Download the library from the `NatNetSDKCrossplatform GitHub repository <https://github.com/whoenig/NatNetSDKCrossplatform>`_.
2. Follow the build instructions in the repository.
3. Add the library to your `LIBRARY_PATH` and `LD_LIBRARY_PATH` variables.

LIBUSB_ERROR_ACCESS (deployment only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you encounter USB access permission issues, change the permissions with:

.. code-block:: bash

   sudo chmod -R 777 /dev/bus/usb/

Next Steps
----------

Once you have successfully set up the project, you can proceed to explore the simulation environment, develop your racing algorithms, and participate in the online competition. Refer to other sections of the documentation for more information on using the project and developing your racing strategies.
