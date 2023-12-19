![banner](https://airo.ugent.be/assets/img/cloth_competition_banner.jpg)

# ICRA 2024 Cloth Competition

> :construction: This repository is under construction. :construction:

Official repository for the [ICRA 2024 Cloth Competition :shirt:](https://airo.ugent.be/cloth_competition/).
This will contain:
* The specification dataset format and expected output.
* A notebook to get started.
* The code for our data collection procedure.

## Installation

```
git clone git@github.com:Victorlouisdg/cloth-competition.git
cd cloth-competition
conda env create -f environment.yaml
pre-commit install
```

**TODO**: Add installation of [airo-mono](https://github.com/airo-ugent/airo-mono/tree/main/airo-camera-toolkit) packages and [linen](https://github.com/Victorlouisdg/linen).

## Releasing
Releasing a new version of `cloth-tools` on PyPi, example:
1. Update the version in `pyproject.toml`.
2. ```git tag -a v0.1.0 -m "cloth-tools v0.1.0"```
3. ```git push origin v0.1.0```