# Autonomous Drone Racing Project Course
<p align="center">
  <img width="460" height="300" src="docs/img/banner.jpeg">
</p>
<sub><sup>AI generated image</sup></sub>

[![Python Version]][Python Version URL] [![Ruff Check]][Ruff Check URL] [![Documentation Status]][Documentation Status URL] [![Tests]][Tests URL]

[Python Version]: https://img.shields.io/badge/python-3.8-blue.svg
[Python Version URL]: https://www.python.org

[Ruff Check]: https://github.com/utiasDSL/lsy_drone_racing/actions/workflows/ruff.yml/badge.svg?style=flat-square
[Ruff Check URL]: https://github.com/utiasDSL/lsy_drone_racing/actions/workflows/ruff.yml

[Documentation Status]: https://readthedocs.org/projects/lsy-drone-racing/badge/?version=latest
[Documentation Status URL]: https://lsy-drone-racing.readthedocs.io/en/latest/?badge=latest

[Tests]: https://github.com/utiasDSL/lsy_drone_racing/actions/workflows/testing.yml/badge.svg
[Tests URL]: https://github.com/utiasDSL/lsy_drone_racing/actions/workflows/testing.yml

## Table of Contents
- [Autonomous Drone Racing Project Course](#autonomous-drone-racing-project-course)
  - [Table of Contents](#table-of-contents)
  - [Documentation](#documentation)
  - [Installation](#installation)
    - [Fork lsy\_drone\_racing](#fork-lsy_drone_racing)
    - [Using conda/mamba](#using-condamamba)
    - [Using Docker](#using-docker)
  - [Difficulty levels](#difficulty-levels)
    - [Switching between configurations](#switching-between-configurations)
  - [The online competition](#the-online-competition)
    - [Signing up for the online competition](#signing-up-for-the-online-competition)
    - [Setting up your GitHub repo for the competition](#setting-up-your-github-repo-for-the-competition)
    - [Submitting your latest iteration](#submitting-your-latest-iteration)
  - [Creating your own controller](#creating-your-own-controller)
  - [Common errors](#common-errors)
    - [GLIBCXX](#glibcxx)
  - [Deployment](#deployment)
    - [Hardware setup](#hardware-setup)
    - [Common errors](#common-errors-1)
      - [libNatNet](#libnatnet)
      - [LIBUSB\_ERROR\_ACCESS](#libusb_error_access)
    - [Deployment on our lab PC](#deployment-on-our-lab-pc)
    - [Fly with the drones](#fly-with-the-drones)
      - [Settings](#settings)
      - [Launch](#launch)


## Documentation
To get you started with the drone racing project, you can head over to our [documentation page](https://lsy-drone-racing.readthedocs.io/en/latest/getting_started/general.html).

## Installation

To run the LSY Autonomous Drone Racing project, you will need 2 repositories:
- [pycffirmware](https://github.com/utiasDSL/pycffirmware/tree/drone_racing) - `drone_racing` branch: A simulator for the on-board controller response of the drones we are using to accurately model their behavior.
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
conda create -n race python=3.8
conda activate race
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
git clone -b drone_racing https://github.com/utiasDSL/pycffirmware.git
cd pycffirmware
git submodule update --init --recursive
sudo apt update
sudo apt install build-essential
./wrapper/build_linux.sh
```

Finally, you can test if the installation was successful by running 

```bash
cd ~/repos/lsy_drone_racing
python scripts/sim.py
```

If everything is installed correctly, this opens the simulator and simulates a drone flying through four gates.

You can also install the extended dependencies with 
```bash
conda activate race
cd ~/repos/lsy_drone_racing
pip install -e .[rl, test]
```
and check if all tests complete with 
```bash
cd ~/repos/lsy_drone_racing
pytest tests
```

### Using Docker
You can also run the simulation with Docker, albeit without the GUI at the moment. To test this, install docker with docker compose on your system, and then run
```bash
docker compose build
docker compose up
```
After building, running the container should produce the following output:

```bash
sim-1  | INFO:__main__:Flight time (s): 8.466666666666667
sim-1  | Reason for termination: Task completed
sim-1  | Gates passed: 4
sim-1  | 
sim-1  | 8.466666666666667
```

## Difficulty levels
The complete problem is specified by a TOML file, e.g. [`level0.toml`](config/level0.toml)

The config folder contains settings for progressively harder scenarios:

|      Evaluation Scenario      | Constraints | Rand. Inertial Properties | Randomized Obstacles, Gates | Rand. Between Episodes |         Notes         |
| :---------------------------: | :---------: | :-----------------------: | :-------------------------: | :--------------------: | :-------------------: |
| [Level 0](config/level0.toml) |   **Yes**   |           *No*            |            *No*             |          *No*          |   Perfect knowledge   |
| [Level 1](config/level1.toml) |   **Yes**   |          **Yes**          |            *No*             |          *No*          |       Adaptive        |
| [Level 2](config/level2.toml) |   **Yes**   |          **Yes**          |           **Yes**           |          *No*          | Learning, re-planning |
| [Level 3](config/level3.toml) |   **Yes**   |          **Yes**          |           **Yes**           |        **Yes**         |      Robustness       |
|                               |             |                           |                             |                        |                       |
|           sim2real            |   **Yes**   |    Real-life hardware     |           **Yes**           |          *No*          |   Sim2real transfer   |

> **Note:** "Rand. Between Episodes" (governed by argument `reseed_on_reset`) states whether randomized properties and positions vary or are kept constant (by re-seeding the random number generator on each `env.reset()`) across episodes

### Switching between configurations
You can choose which configuration to use by changing the `--config` command line option. To e.g. run the example controller on the hardest scenario, you can use the following command

```bash
python scripts/sim.py --config config/level3.toml
```

## The online competition

During the semester, you will compete with the other teams on who's the fastest to complete the drone race. You can see the current standings on the competition page in Kaggle, a popular ML competition website. The results of the competition will **NOT** influence your grade directly. However, it gives you a sense of how performant and robust your approach is compared to others. In addition, the competition is an easy way for you to check if your code is running correctly. If there are errors in the automated testing, chances are your project also doesn't run on our systems. The competition will always use difficulty level 3.

### Signing up for the online competition

To take part in the competition, you first have to create an account on [Kaggle](https://www.kaggle.com/). Next, use this [invite link](https://www.kaggle.com/t/487e7e8777364612ba3f9ea3a6a1ea15) to join the competition, go to the [drone racing competition](https://www.kaggle.com/competitions/lsy-drone-racing-ws24/overview), click on "Rules", and accept the competition conditions. This step is necessary to allow submissions from your account.

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

## Deployment

### Hardware setup

To deploy the controllers on real drones you must install ROS Noetic and the crazyswarm package.

Clone the [crazyswarm repository](https://github.com/USC-ACTLab/crazyswarm) and follow its [build steps](https://crazyswarm.readthedocs.io/en/latest/installation.html).

```bash
cd ~/repos
git clone https://github.com/USC-ACTLab/crazyswarm
...
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

# USB preparation for crazyradio
sudo groupadd plugdev
sudo usermod -a -G plugdev $USER

# Apply changes
sudo udevadm control --reload-rules
sudo udevadm trigger
```

We also need to install the Vicon bridge package to get access to the Vicon positions etc in ROS.
```bash
# Install Vicon bridge nodelet
cd <path/to/catkin_ws>/src/
git clone https://github.com/ethz-asl/vicon_bridge
cd ..
catkin_make
source <path/to/catkin_ws>/devel/setup.bash
```

To start the Vicon bridge by default, you may want to include it in the crazyswarm launchfile.

Optionally, you can also install cfclient to debug issues with the drones and configure IDs etc.
```bash
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

#### LIBUSB_ERROR_ACCESS
Change the USB access permissions with

```sudo chmod -R 777 /dev/bus/usb/```

### Deployment on our lab PC
In order to simplify deployment each team should create their own Robostack environment. 
Please choose descriptive environment names such as ```Team1```.

#### Setting Up Your Python Environment with RoboStack

Follow these steps to set up a Python environment for drone racing projects using RoboStack.

#### Prerequisites
- Ensure ROS is NOT sourced, e.g. in your .bashrc file. Remove any ROS-related sourcing commands temporarily during setup.

- Required Repositories:
  Ensure the following repositories are installed in ```~/repos```: 
    - lsy_drone_racing
    - pycffirmware
    - crazyswarm-import-py11

#### Step-by-Step Setup
##### Create and Activate Python Environment

Run the following commands to create a new environment using mamba (part of the Conda ecosystem):

```bash
export ENV_NAME="<your-group-name>"
mamba create -n $ENV_NAME -c conda-forge -c robostack-staging ros-noetic-desktop python=3.11
mamba activate $ENV_NAME
```

Make sure your environments libraries can be found:

```bash
cd ~/.mamba/envs/$ENV_NAME/etc/conda/activate.d
echo "export LIBRARY_PATH=$LIBRARY_PATH:/home/adr/.mamba/envs/$ENV_NAME/lib" > xlibrary_path.sh
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/adr/.mamba/envs/$ENV_NAME/lib" > xldlibrary_path.sh 
mamba activate $ENV_NAME
```


##### Create your own project folder
```bash
mkdir ~/repos/student_forks/$ENV_NAME
cd ~/repos/student_forks/$ENV_NAME
```

##### Install Dependencies

Navigate to your project directory and install dependencies:

- Clone your fork of the lsy_drone_racing repository and install it:

```bash
cd ~/repos/student_forks/$ENV_NAME
git clone https://github.com/<your-github-username>/lsy_drone_racing.git
cd lsy_drone_racing
pip install -e .
```
- Clone and install the pycffirmare package:

```bash
cd ~/repos/student_forks/$ENV_NAME
git clone -b drone_racing https://github.com/utiasDSL/pycffirmware.git
cd pycffirmware
git submodule update --init --recursive
./wrapper/build_linux.sh
```

- Copy & Install the crazyswarm-import package:

```bash
cp -r ~/repos/crazyswarm-import ~/repos/student_forks/$ENV_NAME
cd ~/repos/student_forks/$ENV_NAME/crazyswarm-import
export CSW_PYTHON=python3
cd ros_ws
rm -rf build devel
cd ..
./build.sh
```

- Copy & build the extras workspace:

```bash
cp -r ~/repos/catkin_ws ~/repos/student_forks/$ENV_NAME
cd ~/repos/student_forks/$ENV_NAME/catkin_ws
rm -rf build devel
catkin_make
```

##### Configure Environment 

Set up environment scripts:

```bash
cd ~/.mamba/envs/$ENV_NAME/etc/conda/activate.d
echo "source $HOME/repos/student_forks/$ENV_NAME/crazyswarm-import/ros_ws/devel/setup.bash" > xsource-crazyswarm.sh
echo "source $HOME/repos/student_forks/$ENV_NAME/catkin_ws/devel/setup.bash --extend" > xsource-extras.sh
```

##### Finalizing Setup
Close all open terminals to ensure that no prior ROS setups interfere with your environment.

Reopen terminals and activate your environment in each terminal:

```bash
mamba activate <your-env-name>
```

### Fly with the drones 

#### Settings
Make sure you are familiar with the configuration files. Not all options are relevant depending on the motion capture setup. For more info, see the [official documentation](https://crazyswarm.readthedocs.io/en/latest/configuration.html#adjust-configuration-files).

The important config files are located in the crazyswarm ROS package:

- [Crazyflies types](https://github.com/USC-ACTLab/crazyswarm/blob/master/ros_ws/src/crazyswarm/launch/crazyflieTypes.yaml) — includes controller properties and marker configurations, etc.
- [In-use Crazyflies](https://github.com/USC-ACTLab/crazyswarm/blob/master/ros_ws/src/crazyswarm/launch/crazyflies.yaml) — includes ID, radio channel, types, etc.

As well as the main launch file [hover_swarm.launch](https://github.com/USC-ACTLab/crazyswarm/blob/master/ros_ws/src/crazyswarm/launch/hover_swarm.launch).

#### Launch
>**Note:** The following should be run within your teams conda environment.

In a terminal, launch the ROS node for the crazyflies. Change the settings in _<path/to/crazyswarm-import/package>/ros_ws/src/crazyswarm/launch/crazyflies.yaml_ as necessary.
```bash
roslaunch crazyswarm hover_swarm.launch
```

In a second terminal:

```bash
python scripts/deploy.py --controller <your_controller.py> --config level3.toml
```

where `<your_controller.py>` implements a controller that inherits from `lsy_drone_racing.control.BaseController`


