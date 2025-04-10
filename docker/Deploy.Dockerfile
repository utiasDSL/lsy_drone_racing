FROM ros:noetic-ros-base-focal

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV CSW_PYTHON=python3
ENV ROS_MASTER_URI=http://host.docker.internal:11311

WORKDIR /home
RUN apt update && apt install python-is-python3 -y
RUN apt-get -y install swig build-essential git python-is-python3 python3-pip libusb-1.0-0-dev
RUN apt-get -y install libpcl-dev
RUN pip install --upgrade pip
# Ignore system version of PyYaml to prevent distutils error "Cannot uninstall 'PyYAML'. It is a distutils installed project"
RUN pip install --ignore-installed "pyyaml>=6.0"
RUN apt-get install -y ros-noetic-tf ros-noetic-tf-conversions

# Install cflib
RUN python -m pip install cflib

# Copy only pyproject.toml first to leverage Docker cache for dependency installation
# This allows us to avoid reinstalling dependencies if only the source code changes
WORKDIR /home/lsy_drone_racing
COPY pyproject.toml ./
# Install dependencies and cache the build step (only rebuilds when pyproject.toml changes) 
RUN pip install build
RUN pip install --no-cache-dir .[test]
# Copy the rest of the application
COPY . .
RUN pip install --no-cache-dir -e .[test]

CMD bash -c "python /home/lsy_drone_racing/scripts/deploy.py"
