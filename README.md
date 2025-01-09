![banner](https://airo.ugent.be/assets/img/cloth_competition_banner.jpg)

# ICRA 2024 Cloth Competition

Official repository for the [ICRA 2024 Cloth Competition :shirt:](https://airo.ugent.be/cloth_competition/).

The dataset (83.0 GB) can be found [on Zenodo](https://zenodo.org/records/14621179).

For general questions about the competition, please use our [Github Discussions page](https://github.com/Victorlouisdg/cloth-competition/discussions).
## Getting started

1. First complete the [installation](#installation) instructions.
2. Then you can run the [getting started notebook 📔](notebooks/01_getting_started.ipynb) to explore the dataset.

## Introduction
This repository contains code to help the participants of the ICRA 2024 Cloth Competition prepare for the competition.
It contains utilities to load and visualize the dataset, but also all the code we use to actually collect the data.
We will also add the code to run the competition and evaluate the performance of the participants.

Some useful starting points:
- 📔 Getting started notebook:  [notebooks/01_getting_started.ipynb](notebooks/01_getting_started.ipynb)
- 📷 Competition observation definition:  [cloth-tools/cloth_tools/dataset/format.py](cloth-tools/cloth_tools/dataset/format.py)
- 👉👕 Manual grasp annotation:  [notebooks/grasping/01_annotation.ipynb](notebooks/grasping/01_annotation.ipynb)
- 📦 Data collection: [scripts/data_collection/09_data_collection_loop.py](scripts/data_collection/09_data_collection_loop.py)



## Installation
The recommended method to install the required dependencies is to use conda.
Make sure you have a version of conda e.g. [miniconda](https://docs.anaconda.com/free/miniconda/) installed.
To make the conda environment creation faster, we recommend configuring the [libmamba solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) first.

Then run the following commands:
```shell
git clone git@github.com:Victorlouisdg/cloth-competition.git
cd cloth-competition
conda env create -f environment.yaml
```
To test your installation:
```shell
conda activate cloth-competition
jupyter notebook notebooks/01_getting_started.ipynb
```

## Development

Use the `environment-dev.yaml` file instead, then install the pre-commit hooks to ensure code quality:
```
pre-commit install
```


### Releasing
Releasing a new version of `cloth-tools` on PyPi, example:
1. Update the version in `pyproject.toml`.
2. ```git tag -a v0.1.0 -m "cloth-tools v0.1.0"```
3. ```git push origin v0.1.0```
