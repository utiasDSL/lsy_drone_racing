FROM python:3.8.10
RUN pip install --upgrade pip
RUN apt update
WORKDIR /home

# Install the drone firmare emulator
RUN apt install swig build-essential git -y
RUN git clone --depth 1 https://github.com/utiasDSL/pycffirmware.git
WORKDIR /home/pycffirmware
RUN git submodule update --init --recursive
# Numpy 2.0 is not compatible with pycffirmware, but would be installed by default
RUN pip install "numpy<2"
RUN ./wrapper/build_linux.sh
RUN rm -rf .git

# Copy only pyproject.toml first to leverage Docker cache for dependency installation
# This allows us to avoid reinstalling dependencies if only the source code changes
WORKDIR /home/lsy_drone_racing
COPY pyproject.toml ./
# Install dependencies 
RUN pip install build
RUN pip install --no-cache-dir .[test,rl]
# Copy the rest of the application
COPY . .
# Install the drone racing package
RUN pip install -e .[test,rl]

ENTRYPOINT ["python", "/home/lsy_drone_racing/scripts/sim.py", "--gui", "False"]