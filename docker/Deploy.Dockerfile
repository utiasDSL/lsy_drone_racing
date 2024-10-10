FROM ros:noetic-ros-base-focal
RUN apt update && apt install python-is-python3 -y
RUN apt-get -y install python3-pip
RUN pip install --upgrade pip
# Ignore system version of PyYaml to prevent distutils error "Cannot uninstall 'PyYAML'. It is a distutils installed project"
RUN pip install --ignore-installed "pyyaml>=6.0"
WORKDIR /home

# Install the drone firmare emulator
RUN apt install swig build-essential git -y
RUN git clone --depth 1 https://github.com/utiasDSL/pycffirmware.git
WORKDIR /home/pycffirmware
RUN git submodule update --init --recursive
# Numpy 2.0 is not compatible with pycffirmware, but would be installed by default
RUN pip install "numpy<2"
RUN ./wrapper/build_linux.sh
RUN rm -rf /home/pycffirmware/.git

# Copy only pyproject.toml first to leverage Docker cache for dependency installation
# This allows us to avoid reinstalling dependencies if only the source code changes
WORKDIR /home/lsy_drone_racing
COPY pyproject.toml ./
# Install dependencies 
RUN pip install build
RUN pip install --no-cache-dir -e .[test,rl]
# Copy the rest of the application
COPY . .

CMD bash -c "source /home/crazyswarm/ros_ws/devel/setup.bash && python /home/lsy_drone_racing/scripts/deploy.py"
