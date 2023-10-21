# Benchmarking / Correctness checking directory

This folder is where benchmarking and correctness checks will happen in.

## Structure

Each benchmark has a `name`, and it's devito definition is stored in a python script `<name>.py`.
Those are:

- wave2d_b
- wave3d_b
- diffusion_2D
- diffusion_3D

Slurm batch files are provided to reproduce the experiments from the paper on ARCHER2 in slurm-jobs.

- `diffusion-X.slurm` / `wave-X.slurm` run the corresponding script distributed among X nodes.

## Usage

Each python script has:
- a `-d` flag, expecting the size of the grid on which to execute the stencil operators.
- a `-nt` flag, expecting the number of iterations to execute.
- a `-so` flag, expecting a space discretization order. we used 2, 4 and 8 in our experiments.

To run the Devito implementation, run the script with `--devito 1`.

To run the xDSL implementation, run the script with `--xdsl 1`.

⚠️ Because of runtime environment incompabilities, please always run each implementation seprately to measure performance.

`setup_wave2d.py` and `setup_wave3d.py` are provided to set up the necessary data to run the corresponding scripts. Run them first with the same `-d <size> -so <space-order>` to generate the expected files.

TODO: put data comparison in-place when using devito and xdsl?

To plot the final values, run the script with `--plot 1`.

## Passing environment variables to devito/omp

Prefixing the `make` command with `NAME=val` will make the variable `NAME` available to all stages in the make file.

Example:

```bash
DEVITO_ARCH=gcc DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG python diffusion_3D.py -d 300 300 300 -nt 300 --xdsl 1
DEVITO_ARCH=gcc DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG python diffusion_3D.py -d 300 300 300 -nt 300 --devito 1
```
