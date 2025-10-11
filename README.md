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

## Introduction

LSY Drone Racing is designed to help you develop and test autonomous drone racing algorithms, both in simulation and in real on the Crazyflie platform. Whether you’re a beginner or an experienced drone enthusiast, this project gives you the tools to start developing and understanding drone racing quickly.

## Documentation

To get you started with the drone racing project, you can head over to our [documentation page](https://lsy-drone-racing.readthedocs.io/en/latest/getting_started/general.html).


## Dependencies

The code of this course is built on other project of the [Learning Systems Lab (LSY)](https://www.ce.cit.tum.de/lsy/home/) at TUM. Feel free to check those out:

- [crazyflow](https://github.com/utiasDSL/crazyflow) - Our drone simulator built for massive simulation speeds while keeping high accuracy and good sim2real performance. 
- [drone-models](https://github.com/utiasDSL/drone-models) - Our collection of accurate drone models used in the simulator or for model-based controllers. 
- [drone-controllers](https://github.com/utiasDSL/drone-controllers) - `main` branch: Controllers of the crazyflie quadrotor.

Note that in the [installation section](#step-by-step-installation), we show how those packages are installed automatically.



## Difficulty levels
The complete problem, from track layout to physics type, is specified by a TOML file, e.g. [`level0.toml`](config/level0.toml). Those files contains multiple difficulty levels from easy (0) to hard (3). The config folder contains settings for those progressively harder scenarios:

|      Evaluation Scenario      | Rand. Inertial Properties | Randomized Obstacles, Gates | Random Tracks |       Notes       |
| :---------------------------: | :-----------------------: | :-------------------------: | :-----------: | :---------------: |
| [Level 0](config/level0.toml) |           *No*            |            *No*             |     *No*      | Perfect knowledge |
| [Level 1](config/level1.toml) |          **Yes**          |            *No*             |     *No*      |     Adaptive      |
| [Level 2](config/level2.toml) |          **Yes**          |           **Yes**           |     *No*      |    Re-planning    |
| [Level 3](config/level3.toml) |          **Yes**          |           **Yes**           |    **Yes**    |  Online planning  |
|         **sim2real**          |  **Real-life hardware**   |           **Yes**           |    **Yes**    | Sim2real transfer |

<!--              | [Bonus](config/multi_level3.toml) |           **Yes**           |    **Yes**    |       *No*        | Multi-agent racing | -->

<!-- > **Warning**: The bonus level has not yet been tested with students. You are **not** expected to solve this level. **Only** touch this if you have a solid solution already and want to take the challenge one level further. -->



## The online competition

During the semester, you will compete with the other teams on who's the fastest to complete the drone race. You can see the current standings on the competition page in Kaggle, a popular ML competition website. The results of the competition will **NOT** influence your grade directly. However, it gives you a sense of how performant and robust your approach is compared to others. In addition, the competition is an easy way for you to check if your code is running correctly. If there are errors in the automated testing, chances are your project also doesn't run on our systems. The competition will always use difficulty level 2. For more information, please refer to the documentation.



