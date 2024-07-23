# Autonomous Drone Racing Project Course
![ADR Banner](docs/img/banner.jpeg)
<sub><sup>AI generated image</sup></sub>

## Table of Contents
- [Autonomous Drone Racing Project Course](#autonomous-drone-racing-project-course)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Fork lsy\_drone\_racing](#fork-lsy_drone_racing)
    - [Using conda/mamba](#using-condamamba)
  - [Difficulty levels](#difficulty-levels)
    - [Switching between configurations](#switching-between-configurations)
  - [The online competition](#the-online-competition)
    - [Signing up for the online competition](#signing-up-for-the-online-competition)
    - [Setting up your GitHub repo for the competition](#setting-up-your-github-repo-for-the-competition)
    - [Submitting your latest iteration](#submitting-your-latest-iteration)
  - [Creating your own controller](#creating-your-own-controller)
  - [Common errors](#common-errors)
  - [Deployment (**NOT IMPORTANT FOR STUDENTS FOR NOW**)](#deployment-not-important-for-students-for-now)
    - [Hardware setup](#hardware-setup)
    - [Common errors](#common-errors)
      - [libNatNet](#libnatnet)
    - [Fly with the drones](#fly-with-the-drones)
      - [Settings](#settings)
      - [Launch](#launch)


## Installation

To run the LSY Autonomous Drone Racing project, you will need 2 repositories:
- [pycffirmware](https://github.com/utiasDSL/pycffirmware) - `main` branch: A simulator for the on-board controller response of the drones we are using to accurately model their behavior
- [lsy_drone_racing](https://github.com/utiasDSL/lsy_drone_racing) - `main` branch: This repository contains the drone simulation, environments, and scripts to simulate and deploy the drones in the racing challenge

### Fork lsy_drone_racing

The first step is to fork the [lsy_drone_racing](https://github.com/utiasDSL/lsy_drone_racing) repository for your own group. This has two purposes: You automatically have your own repository with git version control, and it sets you up for taking part in the online competition and automated testing (see [competition](#the-online-competition)).

If you have never worked with GitHub before, see the [docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) on forking.

### Using conda/mamba

The following assumes that you have a functional installation of either [conda](https://conda.io/projects/conda/en/latest/index.html) or [mamba](https://mamba.readthedocs.io/en/latest/).

First, clone the new fork from your own account and create a new environment with Python 3.8 by running

```bash
mkdir -p ~/repos && cd repos
git clone https://github.com/<YOUR-USERNAME>/lsy_drone_racing.git
conda create -n drone python=3.8
conda activate drone
```

> **Note:** It is important you stick with **Python 3.8**. Yes, it is outdated. Yes, we'd also like to upgrade. However, there are serious issues beyond our control when deploying the code on the real drones with any other version.
Now you can install the lsy_drone_racing package in editable mode from the repository root

```bash
cd ~/repos/lsy_drone_racing
pip install --upgrade pip
pip install -e .
```
In addition, you also need to install the pycffirmware package from source with

```bash
cd ~/repos
git clone https://github.com/utiasDSL/pycffirmware.git
cd pycffirmware
git submodule update --init --recursive
sudo apt update
sudo apt install build-essential
conda install swig
./wrapper/build_linux.sh
```

Finally, you can test if the installation was successful by running 

```bash
cd ~/repos/lsy_drone_racing
python scripts/sim.py
```

If everything is installed correctly, this opens the simulator and simulates a drone flying through four gates.

## Difficulty levels
The complete problem is specified by a YAML file, e.g. [`getting_started.yaml`](config/getting_started.yaml)

The config folder contains settings for progressively harder scenarios:

|         Evaluation Scenario         | Constraints | Rand. Inertial Properties | Randomized Obstacles, Gates | Rand. Between Episodes |         Notes         |
| :---------------------------------: | :---------: | :-----------------------: | :-------------------------: | :--------------------: | :-------------------: |
| [`level0.yaml`](config/level0.yaml) |   **Yes**   |           *No*            |            *No*             |          *No*          |   Perfect knowledge   |
| [`level1.yaml`](config/level1.yaml) |   **Yes**   |          **Yes**          |            *No*             |          *No*          |       Adaptive        |
| [`level2.yaml`](config/level2.yaml) |   **Yes**   |          **Yes**          |           **Yes**           |          *No*          | Learning, re-planning |
| [`level3.yaml`](config/level3.yaml) |   **Yes**   |          **Yes**          |           **Yes**           |        **Yes**         |      Robustness       |
|                                     |             |                           |                             |                        |                       |
|              sim2real               |   **Yes**   |    Real-life hardware     |           **Yes**           |          *No*          |   Sim2real transfer   |

> **Note:** "Rand. Between Episodes" (governed by argument `reseed_on_reset`) states whether randomized properties and positions vary or are kept constant (by re-seeding the random number generator on each `env.reset()`) across episodes

### Switching between configurations
You can choose which configuration to use by changing the `--config` command line option. To e.g. run the example controller on the hardest scenario, you can use the following command

```bash
python scripts/sim.py --config config/level3.yaml
```

## The online competition

During the semester, you will compete with the other teams on who's the fastest to complete the drone race. You can see the current standings on the competition page in Kaggle, a popular ML competition website. The results of the competition will **NOT** influence your grade directly. However, it gives you a sense of how performant and robust your approach is compared to others. In addition, the competition is an easy way for you to check if your code is running correctly. If there are errors in the automated testing, chances are your project also doesn't run on our systems. The competition will always use difficulty level 3.

### Signing up for the online competition

To take part in the competition, you first have to create an account on [Kaggle](https://www.kaggle.com/). Next, use this [invite link](https://www.kaggle.com/t/1a37a7de76c745e29a7d7c61e538d581) to join the competition, go to the [drone racing competition](https://www.kaggle.com/competitions/lsy-drone-racing-ss24/overview), click on "Rules", and accept the competition conditions. This step is necessary to allow submissions from your account.

### Setting up your GitHub repo for the competition

The competition submission to Kaggle is fully automated. However, to make the automation work with your Kaggle account, you first have to save your credentials in GitHub. GitHub offers a way to safely store this information without giving anyone else access to it via its [secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions). Start by opening your account settings on Kaggle, go to the **API** section and click on **Create New Token**. This will download a json file containing two keys: Your account username and an API key. Next, open your lsy_drone_racing GitHub repository in the browser and go to Settings -> Secrets and variables -> Actions

>**Note:** You have to select the repository settings, not your account settings

Here you add two new repository secrets using the information from the json file you downloaded:
- Name: KaggleUsername Secret: INSERT_YOUR_USERNAME
- Name: KaggleKey Secret: INSERT_YOUR_KEY

### Submitting your latest iteration

The whole point of the steps you just took was to set you up to use the GitHub action defined in your repository's [.github folder](/.github/workflows/test_solution.yml). This workflow runs every time you push changes to your repository's `main` or `master` branch. To prevent submitting every iteration of your code, you can create new branches and only merge them into the main branch once you finished your changes. However, we recommend regularly updating your main branch to see how fast you are and if the code runs without problems.

>**Note:** The competition will count your fastest average lap time. If a submission performed worse than a previous iteration, it won't update your standing.

>**Note:** The first time the test runs on your account, it will take longer than usual because it has to install all dependencies in GitHub. We cache this environment, so subsequent runs should be faster.

>**Warning:** Kaggle only accepts 100 submissions per day. While we really hope you don't make 100 commits in a single day, we do mention it just in case. 

Once you have pushed your latest iteration, a GitHub action runner will start testing your implementation. You can check the progress by clicking on the Actions tab of your repository. If the submission fails, you can check the errors. Please let us know if something is not working as intended. If you need additional packages for your project, please make sure to update the [environment.yaml](./environment.yaml) file accordingly. Otherwise, the tests will fail. If you want to get a more detailed summary of your performance, you can have a look at the test output directly:


## Creating your own controller

To implement your own controller, have a look at the [example implementation](./examples/controller.py). We recommend altering the existing example controller instead of creating your own file to not break the testing pipeline. Please also read through the documentation of the controller. You **must not** alter its function signatures. If you encounter problems implementing something with the given interface, contact one of the lecturers.

## Common errors

### GLIBCXX
If you were able to install everything without any issues, but the simulation crashes when running the sim script, you should check the error messages for any errors related to `LIBGL` and `GLIBCXX_3.4.30`. If you don't find any conclusive evidence about what has happened, you might also want to run the simulation in verbose mode for `LIBGL` with

```bash
LIBGL_DEBUG=verbose python scripts/sim.py
```

Next, you should check if your system has the required library installed

```bash
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.30
```

or if it is installed in your conda environment

```bash
strings /path-to-your-conda/envs/your-env-name/lib/libstdc++.so.6 | grep GLIBCXX_3.4.30
```

If neither of those yield any results, you are missing this library and can install it with

```bash
conda install -c conda-forge gcc=12.1.0
```

If the program still crashes and complains about not finding `GLIBCXX_3.4.30`, please update your `LD_LIBRARY_PATH` variable to point to your conda environment's lib folder.

## Deployment (**NOT IMPORTANT FOR STUDENTS FOR NOW**)

### Hardware setup

To deploy the controllers on real drones you must install ROS Noetic and the crazyswarm package.

Create a catkin_ws/src folder if it does not exist already, clone the crazywarm package and build the workspace

**TODO: CREATE WORKING CRAZYSWARM PACKAGE**
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/utiasDSL/crazyswarm.git
mv crazyswarm/* .
cd ..
catkin_make
source devel/setup.bash
```

Next, paste the following block into your terminal
```bash
cat <<EOF | sudo tee /etc/udev/rules.d/99-bitcraze.rules > /dev/null
# Crazyradio (normal operation)
SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="7777", MODE="0664", GROUP="plugdev"
# Bootloader
SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="0101", MODE="0664", GROUP="plugdev"
# Crazyflie (over USB)
SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", MODE="0664", GROUP="plugdev"
EOF
```


```bash
# Install Vicon bridge nodelet
cd <path/to/catkin_ws>/src/
git clone -b vicon-bridge-nodelet git@github.com:utiasDSL/extras.git
cd ..
catkin_make
source <path/to/catkin_ws>/devel/setup.bash

# Install and make crazyflie-firmware-import
cd ~/GitHub
git clone -b dsl-iros-comp-flight git@github.com:utiasDSL/crazyflie-firmware-import.git # other options are `dsl-sim2real-logging-v1`, etc.
cd crazyflie-firmware-import
git submodule update --init --recursive
sudo apt-get install make gcc-arm-none-eabi
make cf2_defconfig # Make the default config file.
make -j 12

# USB preparation for crazyradio
sudo groupadd plugdev
sudo usermod -a -G plugdev $USER
```

```bash
# Apply changes
sudo udevadm control --reload-rules
sudo udevadm trigger

# Flash crazyflie C10 (cf9 in the Vicon objects list)
# Turn the Crazyflie off, then start the Crazyflie in bootloader mode by pressing the power button for 3 seconds. Both the blue LEDs will blink.
cd ~/GitHub/crazyflie-firmware-import/
make cload

# (optional) Install cfclient
sudo apt install libxcb-xinerama0
conda create -n cfclient python=3.7
conda activate cfclient
pip install --upgrade pip # note: we are using a conda python3.7 env
pip install cfclient
conda deactivate
```

### Common errors

#### libNatNet
If libNatNet is missing either during compiling crazyswarm or launching hover_swarm.launch, one option is to manually install it. Download the library from its [github repo](https://github.com/whoenig/NatNetSDKCrossplatform), follow the build instructions, and then add the library to your `LIBRARY_PATH` and `LD_LIBRARY_PATH` variables.

### Fly with the drones 

#### Settings
Make sure you are familiar with the configuration files. Not all options are relevant depending on the motion capture setup. For more info, see the [official documentation](https://crazyswarm.readthedocs.io/en/latest/configuration.html#adjust-configuration-files).

The important config files are located in the crazyswarm ROS package:

**TODO:** Insert correct link to files
- Crazyflies types — includes controller properties and marker configurations, etc.
- In-use Crazyflies — includes ID, radio channel, types, etc.
- All Crazyflies

As well as the launch file and Python script:

- cf_sim2real.launch
- cmdFullStateCFFirmware.py

#### Launch

>**Note:** The following is **NOT** within a conda environment, but has to run directly on the system's Python 3.8 installation. ROS has never heard of these best practices you speak of.

In a terminal, launch the ROS node for the crazyflies. Change the settings in _<path/to/crazyswarm/package>/launch/crazyflies.yaml_ as necessary.
```bash
roslaunch crazyswarm cf_sim2real.launch
```

In a second terminal:

```bash
python scripts/deploy.py --controller <path/to/your/controller.py> --config config/level3.yaml
```

where `<path/to/your/controller.py>` implements a controller that inherits from `lsy_drone_racing.controller.BaseController`


