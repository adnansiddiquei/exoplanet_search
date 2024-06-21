# Exoplanets - Coursework Assignment
## Table of Contents
1. [Environment Set Up](#env)
2. [Running the code](#code)
3. [Understanding the code structure](#code_structure)
3. [Report](#report)

## <a name="env"></a> 1. TEnvironment Set Up
A conda `environment.yml` file is provided in the root of the repository. To create the environment, run
```bash
conda env create -f environment.yml
```
And then activate the environment with
```bash
conda activate as3438_exo
```

## <a name="code"></a> 2. Running the code
The code is split into 2 main scripts, one for Q1 and one for Q2.
To run the code for Q1, run
```bash
python -m src.q1
```
and this will run all the code in the `src/q1/__main__.py` file, and output all plots to the `src/q1/out/` directory.

To run the code for Q2, run
```bash
python -m src.q2
```
and this will run all the code in the `src/q2/__main__.py` file, and output all plots to the `src/q2/out/` directory.

## <a name="code_structure"></a> 3. Understanding the code structure
There are 2 folders, `src/q1` and `src/q2`, each of which contains a `__main__.py` file that contains all the code for the
respective question. The `__main__.py` files utilise functions in the `plotting_utils.py` file to generate plots
and function in the `utils.py` to perform other utility tasks.

Q1 primarily makes use of the `transitleastsquares` package for most of the heavy lifting.

Q2 does not use very many packages, the Quasi-Periodic kernels and GP models are all implemented in pure python in the
`src/q1/utils.py` file, and the `emcee` package is used for the MCMC sampling.

## <a name="report"></a> 4. Report
The report is located at [report/out/main.pdf](report/out/main.pdf).
