# Real-to-Sim Parameter Learning for Deformable Packages Using High-Fidelity Simulators for Robotic Manipulation

[Omey M. Manyar<sup>1,*</sup>](https://omey-manyar.com/), [Hantao Ye<sup>1,*</sup>](https://hantao-ye.github.io/), [Siddharth Mayya<sup>2</sup>](https://www.sidmayya.com/), [Fan Wang<sup>2</sup>](https://faninedinburgh.wixsite.com/mysite-1), [Satyandra K. Gupta<sup>1</sup>](https://sites.usc.edu/skgupta/)

<sup>1</sup>University of Southern California, <sup>2</sup>Amazon Robotics

<sup>*</sup>Equal contribution, Listed Alphabetically

**[Project Page](https://sites.google.com/usc.edu/deformable-sim/) | [Video](https://youtu.be/7sGZ8UyUNd0) | [Paper](https://sites.google.com/usc.edu/deformable-sim/paper)**

This repository is the official implementation of the paper. This code is intended for reproduction purposes only. Current implementation does not support extensions. The objective of this repository is to provide the reader with the implementation details of the simulation proposed in the paper.

- [Real-to-Sim Parameter Learning for Deformable Packages Using High-Fidelity Simulators for Robotic Manipulation](#real-to-sim-parameter-learning-for-deformable-packages-using-high-fidelity-simulators-for-robotic-manipulation)
  - [Environment Setup](#environment-setup)
    - [Pre-requirements](#pre-requirements)
    - [Step 1: Install PDM](#step-1-install-pdm)
    - [Step 2: Clone the Repository](#step-2-clone-the-repository)
    - [Step 3: Create a Virtual Environment](#step-3-create-a-virtual-environment)
    - [Step 4: Activate the Virtual Environment](#step-4-activate-the-virtual-environment)
    - [Step 5: Visulization Software](#step-5-visulization-software)
  - [Simulation Pipeline](#simulation-pipeline)
    - [Evaluation](#evaluation)
  - [Parallel Pipeline](#parallel-pipeline)

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
git clone https://github.com/RROS-Lab/deformable-sim.git
cd deformable-sim
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

Firstly, select the created virtual envrionment `0` using `pdm use`

```shell
$ pdm use

Please enter the Python interpreter to use
 0. cpython@3.12 (/path/to/deformable-sim/.venv/bin/python)
 1. cpython@3.12 (/path/to/deformable-sim/.venv/bin/python3.12)
 2. cpython@3.12 (/home/user/miniconda3/bin/python3.12)
 3. cpython@3.12 (/usr/bin/python3.12)
 4. cpython@3.12 (/usr/bin/python)
```

To activate the virtual environment and install dependencies, run:

```shell
eval $(pdm venv activate in-project)
pdm install
```

All necessary dependencies will be installed after running the command above.

### Step 5: Visulization Software

We used [Omniverse Kit](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/kit_overview.html) for visualizing `.usd` files, use the [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/omniverse/collections/kit) to download the software.

After extracted compressed file, run the following inside the compressed folder to start the visualizer:

```shell
./omni.app.editor.base.sh
```

## Simulation Pipeline

The main environment we are using is [package_parameter_optim](./src/envs/package_parameter_optim.py):

```shell
$ python -m src.envs.package_parameter_optim --help
usage: package_parameter_optim.py [-h] [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR] [--iterations ITERATIONS] [--train_test_ratio TRAIN_TEST_RATIO] [--init_points INIT_POINTS]
                                  [--eval EVAL] [--benchmark BENCHMARK]

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory containing the data files. (default: data)
  --output_dir OUTPUT_DIR
                        Directory to save the output files. (default: result)
  --iterations ITERATIONS
                        Number of iterations for sampling. (default: 1000)
  --train_test_ratio TRAIN_TEST_RATIO
                        Ratio of training data to test data. (default: 0.8)
  --init_points INIT_POINTS
                        Number of initial registration points for Bayesian optimization. (default: 5)
  --eval EVAL           Evaluation mode. (default: False)
  --benchmark BENCHMARK
                        Benchmark mode. (default: False)
```

For running parameter identification based on the [experiment data](./data/), run:

```shell
python -m src.envs.package_parameter_optim --iteration 200 # e.g. 200 iterations
```

After it has been finished, `./result/package_parameter_sampling/best_params.csv` storing the best parameters will be generated.

### Evaluation

For detailed evaluation, `.usd` files for each iteration will be stored in `./result/package_parameter_sampling/sampling/*` and optimizaiton history will be reported in `./result/package_parameter_sampling/package_parameter.csv`.

We also finished several scripts for ease of evaluation, run the following:

```shell
python ./scripts/best_parameter.py
```

to generate `best.csv`, which will be used for evaluation and benchmark in the environment.

**Evaluation** mode will run under optimized and perturbed parameters through all *test* and *train* trajectories.  **Benchmark** mode will test optimized parameters under *test* trajectories with various number of particles to report loss and FPS difference.

```shell
python -m src.envs.package_parameter_optim --eval True
python -m src.envs.package_parameter_optim --benchmark True
```

After they have finished, use another script to print statistical report in the terminal:

```shell
python ./scripts/report.py --mode evaluation
python ./scripts/report.py --mode benchmark
```

`train_{idx}/test_{idx}` in evaluation will print losses through *train/test* trajectories under `10*{idx}%` perturbation from optimized parameters. `test_{idx}` in benchmark will print losses through *test* trajectories by adding `{idx}` particles in between sampled points for package simulation.

## Parallel Pipeline

TBA
