# Python Environment Setup

## Install Conda

Install miniconda following instruction from the [official website](https://docs.conda.io/en/latest/miniconda.html).


## Create a new environment of Python 3.9

```bash
conda create -n summer_camp python=3.9
```

## Activate the environment

```bash
conda activate summer_camp
```

## Install required packages

```bash
pip install --no-cache-dir -r requirements.txt
```

## Deactivate the environment

```bash
conda deactivate
```

## Useful commands for managing the environment

- Remove the environment

```bash
conda remove -n summer_camp --all
```

- Update the environment

```bash
conda update -n summer_camp --all
```

- List all environments

```bash
conda env list
```

- List all packages in the environment

```bash
conda list
```

- Export the environment

```bash
conda env export > environment.yml
```

- Create an environment from the file

```bash
conda env create -f environment.yml
```

