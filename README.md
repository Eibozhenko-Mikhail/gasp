# Generalzed Adaptive Skill Prior Meta-Reinforcement Learning

<p align="center">
<img src="docs/resources/Teaser.png" width="800">
</p>
</img>

This page is dedicated to implementation of "**Generalized Adaptive Skill Prior Meta-Reinforcement Learning**" algorithm in the context of final Master Thesis for chair of Robotics, Artificial Intelligence and Real-time Systems at TUM School of Computation, Information and Technology authored by Mikhail Eibozhenko.

## Acknowledges

The proposed solution is based on two frameworks, refer to pages of algorithms mentioned below for more information:
- [SPiRL](https://github.com/clvrai/spirl) - Skill-Prior framework for Reinforcement Learning acceleration
- [DIVA](https://github.com/Ghiara/DIVA) - VAE-based non-parametric Latent space clustering 


## Requirements

- python 3.7
- bnpy 1.7.0
- mujoco 2.0
- pytorch-lightning 1.9.5
- Ubuntu 18.04

## Installation Instructions

### 1. Create and activate a virtual environment, install all requirements
```
# Setup the environment
cd SPIRL_DPMM
pip3 install virtualenv
virtualenv -p $(which python3) ./venv
source ./venv/bin/activate

# Install dependencies and packages
pip3 install -r requirements.txt
pip3 install -e .
```

### 2. Define environment variables to specify the root experiment and data directories
```
# Experiments folder stores trained models
# Data folder stores external data libraries
mkdir ./experiments
mkdir ./data
export EXP_DIR=./experiments
export DATA_DIR=./data
```

### 3. Install the Fork of D4RL benchmark

Follow the [D4RL Fork link](https://github.com/kpertsch/d4rl) and install the fork according to instructions.
This fork includes new key 'completed_tasks' in Kitchen environment and **neccessary for correct RL phase**.

### 4. Log in to WandB to track results

[WandB](https://www.wandb.com/) is used for **logging the training process**. Before running any of the commands below, 
create an account and then change the WandB entity and project name at the top of [train.py](spirl/train.py) and
[rl/train.py](spirl/rl/train.py) to match your account.

## Main Commands

### Training Generalized Adaptive Skill Prior

To train a **Generalized Adaptive Skill Prior** model, run:
```
python3 spirl/train.py --path=spirl/configs/skill_prior_learning/kitchen/spirl_DPMM_h_cl --val_data_size=160
```

### Training GASP Meta-RL

After GASP model is trained, to train **GASP Meta-RL** agent on the kitchen environment, run:
```
python3 spirl/rl/train.py --path=spirl/configs/hrl/kitchen/spirl_cl_DPMM --seed=0 --prefix=GASP_kitchen_seed0
```

### Visualizing learned DPMM distribution

After GASP model is trained, vizualize the latent distribution compared with original gaussian with
```
python3 analysis/DPMM_vis.py
```

### Visualizing Inference

Another visualization tool implemented allows to project encoded inference onto learned DPMM space - run
```
python3 analysis/SD_Inference.py
```

## Starting to Modify the Code

### Modifying the hyperparameters



## Detailed Code Structure Overview
```
spirl
  |- components            # reusable infrastructure for model training
  |    |- base_model.py    # basic model class that all models inherit from
  |    |- checkpointer.py  # handles storing + loading of model checkpoints
  |    |- data_loader.py   # basic dataset classes, new datasets need to inherit from here
  |    |- evaluator.py     # defines basic evaluation routines, eg top-of-N evaluation, + eval logging
  |    |- logger.py        # implements core logging functionality using tensorboardX
  |    |- params.py        # definition of command line params for model training
  |    |- trainer_base.py  # basic training utils used in main trainer file
  |
  |- configs               # all experiment configs should be placed here
  |    |- data_collect     # configs for data collection runs
  |    |- default_data_configs   # defines one default data config per dataset, e.g. state/action dim etc
  |    |- hrl              # configs for hierarchical downstream RL
  |    |- rl               # configs for non-hierarchical downstream RL
  |    |- skill_prior_learning   # configs for skill embedding and prior training (both hierarchical and flat)
  |
  |- data                  # any dataset-specific code (like data generation scripts, custom loaders etc)
  |- models                # holds all model classes that implement forward, loss, visualization
  |- modules               # reusable architecture components (like MLPs, CNNs, LSTMs, Flows etc)
  |- rl                    # all code related to RL
  |    |- agents           # implements core algorithms in agent classes, like SAC etc
  |    |- components       # reusable infrastructure for RL experiments
  |        |- agent.py     # basic agent and hierarchial agent classes - do not implement any specific RL algo
  |        |- critic.py    # basic critic implementations (eg MLP-based critic)
  |        |- environment.py    # defines environment interface, basic gym env
  |        |- normalization.py  # observation normalization classes, only optional
  |        |- params.py    # definition of command line params for RL training
  |        |- policy.py    # basic policy interface definition
  |        |- replay_buffer.py  # simple numpy-array replay buffer, uniform sampling and versions
  |        |- sampler.py   # rollout sampler for collecting experience, for flat and hierarchical agents
  |    |- envs             # all custom RL environments should be defined here
  |    |- policies         # policy implementations go here, MLP-policy and RandomAction are implemented
  |    |- utils            # utilities for RL code like MPI, WandB related code
  |    |- train.py         # main RL training script, builds all components + runs training
  |
  |- utils                 # general utilities, pytorch / visualization utilities etc
  |- train.py              # main model training script, builds all components + runs training loop and logging
```

The general philosophy is that each new experiment gets a new config file that captures all hyperparameters etc. so that experiments
themselves are version controllable.

## Datasets

|Dataset        | Link         | Size |
|:------------- |:-------------|:-----|
| Maze | [https://drive.google.com/file/d/1pXM-EDCwFrfgUjxITBsR48FqW9gMoXYZ/view?usp=sharing](https://drive.google.com/file/d/1pXM-EDCwFrfgUjxITBsR48FqW9gMoXYZ/view?usp=sharing) | 12GB |
| Block Stacking |[https://drive.google.com/file/d/1VobNYJQw_Uwax0kbFG7KOXTgv6ja2s1M/view?usp=sharing](https://drive.google.com/file/d/1VobNYJQw_Uwax0kbFG7KOXTgv6ja2s1M/view?usp=sharing)| 11GB|
| Office Cleanup | [https://drive.google.com/file/d/1yNsTZkefMMvdbIBe-dTHJxgPIRXyxzb7/view?usp=sharing](https://drive.google.com/file/d/1yNsTZkefMMvdbIBe-dTHJxgPIRXyxzb7/view?usp=sharing)| 170MB |

You can download the datasets used for the experiments in the paper with the links above. 
To download the data via the command line, see example commands [here](spirl/data/).

If you want to generate more data 
or make other modifications to the data generating procedure, we provide instructions for regenerating the 
`maze`, `block stacking` and `office` datasets [here](spirl/data/).


## Citation
If you find this work useful in your research, please consider citing:
```
@inproceedings{pertsch2020spirl,
    title={Accelerating Reinforcement Learning with Learned Skill Priors},
    author={Karl Pertsch and Youngwoon Lee and Joseph J. Lim},
    booktitle={Conference on Robot Learning (CoRL)},
    year={2020},
}
```

## Acknowledgements
The model architecture and training code builds on a code base which we jointly developed with [Oleh Rybkin](https://www.seas.upenn.edu/~oleh/) for our previous project on [hierarchial prediction](https://github.com/orybkin/video-gcp).

We also published many of the utils / architectural building blocks in a stand-alone package for easy import into your 
own research projects: check out the [blox](https://github.com/orybkin/blox-nn) python module. 


## Troubleshooting

### Missing key 'completed_tasks' in Kitchen environment
Please make sure to install [our fork](https://github.com/kpertsch/d4rl) of the D4RL repository, **not** the original D4RL repository. We made a few small changes to the interface, which e.g. allow us to log the reward contributions for each of the subtasks separately.




