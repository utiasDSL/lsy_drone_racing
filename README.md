# Autonomous Drone Racing Project Course
<p align="center">
  <img width="460" height="300" src="docs/img/banner.jpeg">
</p>
<sub><sup>AI generated image</sup></sub>

[![Python Version]][Python Version URL] [![Ruff Check]][Ruff Check URL] [![Documentation Status]][Documentation Status URL] [![Tests]][Tests URL]

[Python Version]: https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue
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
  - [Required Packages - Overview](#required-packages---overview)
      - [In Simulation](#in-simulation)
      - [On Hardware](#on-hardware)
  - [Step-by-Step Installation](#step-by-step-installation)
    - [Fork lsy\_drone\_racing](#fork-lsy_drone_racing)
    - [Setting up your environment](#setting-up-your-environment)
      - [Pixi package manager (Recommended)](#pixi-package-manager-recommended)
      - [Micromamba package manager (Not recommended)](#micromamba-package-manager-not-recommended)
      - [Simulation \& Hardware on our Lab PC (If Necessary)](#simulation--hardware-on-our-lab-pc-if-necessary)
      - [Simulation only (Not Recommended)](#simulation-only-not-recommended)
    - [Installation](#installation)
      - [Clone repository](#clone-repository)
      - [Install simulation (developing \& testing controllers)](#install-simulation-developing--testing-controllers)
      - [Install deployment (deploy controller to real drones)](#install-deployment-deploy-controller-to-real-drones)
      - [Install Acados (necessary for MPC approaches in sim \& real)](#install-acados-necessary-for-mpc-approaches-in-sim--real)
    - [Development](#development)
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
      - [LIBUSB\_ERROR\_ACCESS](#libusb_error_access)
      - [Settings](#settings)
      - [Launch](#launch)



## Documentation

To get you started with the drone racing project, you can head over to our [documentation page](https://lsy-drone-racing.readthedocs.io/en/latest/getting_started/general.html).


## Required Packages - Overview

#### In Simulation

To run the LSY Autonomous Drone Racing project course in simulation, you will need the following repositories:
- [lsy_drone_racing](https://github.com/utiasDSL/lsy_drone_racing) - `main` branch: This repository contains the environments and scripts to simulate and deploy the drones in the racing challenge.
- [crazyflow](https://github.com/utiasDSL/crazyflow) - `main` branch: This repository constains the drone simulation.
- [drone-models](https://github.com/utiasDSL/drone-models) - `main` branch: Dynamics Models of the crazyflie quadrotor.
- [drone-controllers](https://github.com/utiasDSL/drone-controllers) - `main` branch: Controllers of the crazyflie quadrotor.

Note that in the [installation section](#step-by-step-installation), we show how those packages are installed automatically.

#### On Hardware

To run the LSY Autonomous Drone Racing project in the lab on real hardware, you need three additional repositories to the ones required for the simulation:
- [motion_capture_tracking](https://github.com/utiasDSL/motion_capture_tracking) - `ros2` branch: This repository is a ROS 2 package that receives data from our VICON motion capture system and publishes it under the `\tf` topic via ROS2. 
- [drone-estimators](https://github.com/utiasDSL/drone-estimators) - `main` branch: Estimators to accurately predict the drone state based on the vicon measurements.

## Step-by-Step Installation

### Fork lsy_drone_racing

The first step is to fork the [lsy_drone_racing](https://github.com/utiasDSL/lsy_drone_racing) repository for your own group. This has two purposes: You automatically have your own repository with git version control, and it sets you up for taking part in the online competition and automated testing (see [competition](#the-online-competition)).

If you have never worked with GitHub before, see the [docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) on forking.

### Setting up your environment

#### Pixi package manager (Recommended)

We recommend using [Pixi](https://pixi.sh) to manage dependencies and virtual environments for this project. Pixi creates a dedicated .pixi directory in the project root, which contains the isolated virtual environment. Installed packages are cached globally under ~/.cache/rattler/ to speed up subsequent environment setups across projects.

To activate the environment, simply run "pixi shell", or "pixi shell -e <environment_name>" for specific sub-environments. On the first invocation, Pixi will automatically resolve and install all required dependencies.

> **Note:** In pixi shell, you can still use pip to install any packages you need.

To [install pixi](https://pixi.sh/latest/installation/), run:
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

#### Micromamba package manager (Not recommended)

Alternatively, [RoboStack](https://robostack.github.io) and [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for this. Robostack lets you install your favorite ROS version independent of you OS. It builds on top of conda/mamba to do this. We do not recommend this, since Mamba/Conda environments have the tendency to be leaky and share some system-wide packages. In our experience, this will lead to problems, which is why this project is optimized to use with pixi. If you want to use micromamba anyway, we can't guarantee support. In that case, please follow the [Robostack Getting Started](https://robostack.github.io/GettingStarted.html) in order to create a ROS2 Jazzy Environment on Python3.11 using micromamba. 

#### Simulation & Hardware on our Lab PC (If Necessary)

We provide a PC in the Lab on which you are allowed to run your controllers during deployment. Please create a new user for each team and follow the instructions below.

#### Simulation only (Not Recommended)

If you only want to run the simulation, you can use your favorite conda/mamba/venv to install the packages. However, you will need a working installation of ROS2 if you want to deploy your controller on the real drone.


### Installation

#### Clone repository

First, clone the new fork from your own account and create a new environment by running

```bash
mkdir -p ~/repos && cd repos
git clone https://github.com/<YOUR-USERNAME>/lsy_drone_racing.git
cd lsy_drone_racing
```

#### Install simulation (developing & testing controllers)

Stay in the repository and run the following command to activate your pixi shell with the default (sim) environment:

```bash
pixi shell
```
> **Note:** This package currently depends on a prerelease version of [scipy](https://github.com/scipy/scipy), which needs to be built from source. This might take more than 10 minutes on older hardware.

> **Note:** By running the commands above, you will have **acados** installed by default. This might cause the terminal to freeze for several minutes.

> **Note:** To leave a pixi shell, press **ESC** or **Ctrl+D**. Make sure to leave the shell before you activate another shell.

To speed up simulation with GPU (optional), run:
```bash
pixi shell -e gpu
```

Finally, you can test if the installation was successful by running 

```bash
cd ~/repos/lsy_drone_racing
python scripts/sim.py
```

If everything is installed correctly, this opens the simulator and simulates a drone flying through four gates.

(Optional) If you want to train RL policies, we recommend you to use [PyTorch](https://pytorch.org/). After choosing the correct hardware setup, you can install torch based on the pip command shown [here](https://pytorch.org/get-started/locally/).

(Optional) You can also run the tests by directly running either 
```bash
pixi run -e tests tests -v
```
or by first activating the correct environment
```bash
pixi shell -e tests
cd ~/repos/lsy_drone_racing
pytest tests
```
#### Install deployment (deploy controller to real drones)

Stay in the repository, run:

```bash
pixi shell -e deploy
```

This will automatically create a ros2 workspace, clone the motion_capture_tracking package, and build it. Thus, the terminal might freeze for 1 minute after regular installation.

Then we still have to install extra packages with pip, **stay in the *deploy* shell**, run one of the commands below:

```bash
pip install -e .[sim,deploy]
pip install -e .[sim,gpu,deploy]
```

> **Note:** This package currently depends on a prerelease version of [scipy](https://github.com/scipy/scipy), which needs to be built from source. This might take more than 10 minutes on older hardware.

Test your installation: For this to work you have to be in the lab and be connected to our local network. 
```
# Check connection to the vicon PC
ping 10.157.163.191
```

If this works, stay in the *deploy* shell and run the motion capture tracking node, if there are valid elements in the motion capture area, you should be able to see them in rviz.
```
ros2 launch motion_capture_tracking launch.py
```
If you already have your drone placed inside the motion capture area, you can start another *deploy* shell and run the estimator node. Please check the actual DEC number on the drone, or the name shown in rviz.
```bash
python -m drone_estimators.ros_nodes.ros2_node --drone_name cf52
```
If this works, you should be able to see frequency information in terminal. 

To send commands to the Crazyflie drones, we need to prepare the USB port for the Crazyradio by executing the following block. Ask the TA to help you with sudo rights.
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

Now you are ready to deploy your controller on real drones. Start a new *deploy* shell, run:
```bash
python scripts/deploy.py --config level2.toml --controller <your_controller.py>
```
> **Note:** Be carefull with flying the drone! Make sure to kill the process (**Ctrl+C**) immediately when your controller is not stable.


#### Install Acados (necessary for MPC approaches in sim & real)
[Acados](https://docs.acados.org/index.html) is an Optimal Control Framework that can be used to control the quadrotor using a Model Predictive Controller.
We prepared an automatic script to install and activate acados in all environments. If something does not work out of the box, we refer the reader to the [official installation guide](https://docs.acados.org/installation/).


### Development
If you want to do more in-depth development or understand the used packages ([crazyflow](https://github.com/utiasDSL/crazyflow), [drone-models](https://github.com/utiasDSL/drone-models), [drone-controllers](https://github.com/utiasDSL/drone-controllers), [drone-estimators](https://github.com/utiasDSL/drone-estimators)) better, you can fork and install all of those packages separately in editable mode. If you find bugs or other have improvements, feel free to submit a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) or [create an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue). The installation procedure is the same for all packages:
```bash
cd ~/repos/lsy_drone_racing
pixi shell
cd ~/repos
git clone https://github.com/<YOUR-USERNAME>/crazyflow.git
cd ~/repos/crazyflow
pip install -e .
cd ~/repos/lsy_drone_racing
```

After installation, you can change files in the cloned repository and the changes will directly affect your environment. 

### Using Docker
You can also run the *simulation* with Docker, albeit without the GUI at the moment. To test this, install docker with docker compose on your system, and then run
```bash
docker compose --profile sim build
docker compose --profile sim up
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

|      Evaluation Scenario      |     Rand. Inertial Properties     | Randomized Obstacles, Gates | Random Tracks |       Notes       |
| :---------------------------: | :-------------------------------: | :-------------------------: | :-----------: | :---------------: |
| [Level 0](config/level0.toml) |               *No*                |            *No*             |     *No*      | Perfect knowledge |
| [Level 1](config/level1.toml) |              **Yes**              |            *No*             |     *No*      |     Adaptive      |
| [Level 2](config/level2.toml) |              **Yes**              |           **Yes**           |     *No*      |    Re-planning    |
| [Level 3](config/level3.toml) |              **Yes**              |           **Yes**           |    **Yes**    |  Online planning  |
|         **sim2real**          |      **Real-life hardware**       |           **Yes**           |    **Yes**    | Sim2real transfer |
|             <!--              | [Bonus](config/multi_level3.toml) |           **Yes**           |    **Yes**    |       *No*        | Multi-agent racing | --> |

<!-- > **Warning**: The bonus level has not yet been tested with students. You are **not** expected to solve this level. **Only** touch this if you have a solid solution already and want to take the challenge one level further. -->

### Switching between configurations
You can choose which configuration to use by changing the `--config` command line option. To e.g. run the example controller on the hardest simulation scenario, you can use the following command

```bash
python scripts/sim.py --config config/level2.toml
```

## The online competition

During the semester, you will compete with the other teams on who's the fastest to complete the drone race. You can see the current standings on the competition page in Kaggle, a popular ML competition website. The results of the competition will **NOT** influence your grade directly. However, it gives you a sense of how performant and robust your approach is compared to others. In addition, the competition is an easy way for you to check if your code is running correctly. If there are errors in the automated testing, chances are your project also doesn't run on our systems. The competition will always use difficulty level 2.

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

or if it is installed in your mamba environment

```bash
strings /<PATH-TO-YOUR-MAMBA-ENV>/lib/libstdc++.so.6 | grep GLIBCXX_3.4.30
```

If neither of those yield any results, you are missing this library and can install it with

```bash
micromamba install -c conda-forge gcc=12.1.0
```

If the program still crashes and complains about not finding `GLIBCXX_3.4.30`, please update your `LD_LIBRARY_PATH` variable to point to your mamba environment's lib folder.

### Deployment

#### LIBUSB_ERROR_ACCESS
Change the USB access permissions with

```sudo chmod -R 777 /dev/bus/usb/```

#### Settings
Make sure your drone is selected on the vicon system, otherwise it will not be tracked.

You will have to modify the following config files before liftoff:

Please modify either `~/repos/lsy_drone_racing/config/level2.toml` or create your own custom config file to include the correct drone id.

#### Launch
You will need a total of three terminal windows for deploying your controller on the real drone:

Terminal 1 launches the motion_capture_tracking an ensures that the position of all vicon objects are published via ros2:
```
pixi shell -e deploy
ros2 launch motion_capture_tracking launch.py
```

Terminal 2 starts the estimator: 
```
pixi shell -e deploy
python -m drone_estimators.ros_nodes.ros2_node --drone_name cf52
```

Terminal 3 starts the deployment of the controller: 
```
pixi shell -e deploy
python scritps/deploy.py --config level2.toml --controller <your_controller.py>
```

where `<your_controller.py>` implements a controller that inherits from `lsy_drone_racing.control.BaseController`


