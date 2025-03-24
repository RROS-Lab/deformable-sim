# Real-to-Sim Parameter Learning for Deformable Packages Using High-Fidelity Simulators for Robotic Manipulation

[Omey M. Manyar<sup>1,*</sup>](https://omey-manyar.com/), [Hantao Ye<sup>1,*</sup>](https://hantao-ye.github.io/), [Siddharth Mayya<sup>2</sup>](https://www.sidmayya.com/), [Fan Wang<sup>2</sup>](https://faninedinburgh.wixsite.com/mysite-1), [Satyandra K. Gupta<sup>1</sup>](https://sites.usc.edu/skgupta/)

<sup>1</sup>University of Southern California, <sup>2</sup>Amazon Robotics

<sup>*</sup>Equal contribution, Listed Alphabetically

**[Project Page](https://sites.google.com/usc.edu/deformable-sim/) | Video | Paper**

This repository is the official implementation of the paper. This code is intended for reproduction purposes only. Current implementation does not support extensions. The objective of this repository is to provide the reader with the implementation details of the simulation proposed in the paper.

- [Real-to-Sim Parameter Learning for Deformable Packages Using High-Fidelity Simulators for Robotic Manipulation](#real-to-sim-parameter-learning-for-deformable-packages-using-high-fidelity-simulators-for-robotic-manipulation)
  - [Environment Setup](#environment-setup)
    - [Pre-requirements](#pre-requirements)
    - [Step 1: Install PDM](#step-1-install-pdm)
    - [Step 2: Clone the Repository](#step-2-clone-the-repository)
    - [Step 3: Create a Virtual Environment](#step-3-create-a-virtual-environment)
    - [Step 4: Activate the Virtual Environment](#step-4-activate-the-virtual-environment)

## Environment Setup

To simplify the process of setting up the development environment, we use **[PDM](https://pdm-project.org/en/latest/)** for Python package and virtual environment management.

### Pre-requirements

- Ubuntu >= 22.04
- Python >= 3.12
- Anaconda/Miniconda (For virtualenv creation)

### Step 1: Install PDM

To install PDM, run the following command:

```shell
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

### Step 2: Clone the Repository

Clone the project repository and navigate into the project folder:

```shell
git clone https://github.com/RROS-Lab/IROS2024-Bin-Packing.git
cd IROS2024-Bin-Packing
```

### Step 3: Create a Virtual Environment

Next, create a Python 3.12 virtual environment using PDM and select conda as backend:

```shell
pdm venv create --with conda 3.12
```

To verify the virtual environment was created successfully, use:

```shell
pdm venv list
```

You should see output like:

```shell
Virtualenvs created with this project:

*  in-project: /path/to/deformable-sim/.venv
```

Here, `in-project` is the default name of the virtual environment. If you'd like to specify a custom name for the environment, use:

```shell
pdm venv create --with conda --name my-env-name 3.12
```

### Step 4: Activate the Virtual Environment

To activate the virtual environment and install dependencies, run:

```shell
eval $(pdm venv activate in-project)
pdm install
```

All necessary dependencies will be installed after running the command above.

