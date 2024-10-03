# SEAL (Super Efficient Acados Learning)

## Introduction

SEAL provides tools for learning optimal control policies using learning
methodologies Imitation learning (IL) and Reinforcement Learning (RL) to enhance
Model Predictive Control (MPC) policies. It is built on top of
[acados](https://docs.acados.org/index.html) and [casadi](https://web.casadi.org/).

## Installation

### Dependencies

SEAL requires the following dependencies that need to be installed separately:

- [casadi](https://web.casadi.org/) for symbolic computations
- [acados](https://docs.acados.org/index.html) for generating OCP solvers

### Installation steps

Create python virtual environment and install the (editable) package with the following commands:

``` bash
    sudo pip3 install virtualenv
    cd <PATH_TO_VENV_DIRECTORY>
    virtualenv seal_venv --python=/usr/bin/python3
    source seal_venv/bin/activate
```

Install the minimum:

``` bash
    python -m pip install -e .
```

or install with optional dependencies (e.g. for testing and linting):

``` bash
    python -m pip install -e .[test,lint]
```

## Usage

Try an example with the following command:

``` bash
    python examples/linear_system_mpc.py
```
