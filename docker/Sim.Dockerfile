FROM python:3.11.12
RUN pip install --upgrade pip
RUN apt update
WORKDIR /home

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # python3.11 \
    # python3-pip \
    # python3.11-venv \
    # python3.11-dev \
    bash-completion \
    build-essential \
    curl \
    cmake \
    git \
    ssh \
    sudo \
    mesa-utils \
    wget \
    htop \
    tmux \
    nano \
    gfortran \
    libopenblas-dev \ 
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Copy only pyproject.toml first to leverage Docker cache for dependency installation
# This allows us to avoid reinstalling dependencies if only the source code changes
WORKDIR /home/lsy_drone_racing
COPY pyproject.toml ./
# Install dependencies and cache the build step (only rebuilds when pyproject.toml changes) 
RUN pip install build
RUN pip install --no-cache-dir .[tests,sim,gpu]
# Copy the rest of the application
COPY . .
RUN pip install -e .[tests,sim,gpu]
# Install acados
ENV PIXI_PROJECT_ROOT="/home/lsy_drone_racing"
RUN bash tools/setup_acados.sh

ENTRYPOINT ["python", "/home/lsy_drone_racing/scripts/sim.py", "-r", "False"]