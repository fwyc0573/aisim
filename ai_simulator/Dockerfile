FROM ubuntu:18.04

# python version is related to tensorflow version
ARG python=3.7
ENV PYTHON_VERSION=${python}

# install the dependencies
RUN apt-get update && apt-get install --no-install-recommends --allow-downgrades -y \
    software-properties-common \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip

# set softlink of python3 and python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Install python packages
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install setuptools && \
    python3 -m pip install flake8 \
                           pytest \
                           tensorflow==1.15 \
                           protobuf \
                           bitmath \
                           humanreadable \
                           PyYAML

# Download simulator_benchmark program
COPY simulator_benchmark /simulator_benchmark

# Download SuperScaler project using git, or use local file
# RUN git clone https://github.com/microsoft/SuperScaler.git
COPY SuperScaler /SuperScaler

# Set PYTHONPATH to local file or install SuperScaler package directly
# RUN cd /SuperScaler && python -m pip install .
ENV PYTHONPATH "${PYTONPATH}:/SuperScaler/src"

# Automately run benchmark if run docker directly without command
CMD python /simulator_benchmark/simulator_benchmark.py
