# Temporal blocking of finite-difference stencil operators with sparse "off-the-grid" sources
## This repo contains code used to derive the results os the paper

Contact: g.bisbas18@imperial.ac.uk for more info and help regarding this experiments
Readme.md is under construction.


# Prerequisites:
- A working devito installation https://www.devitoproject.org/devito/download.html
  
# Steps:

```
cd devito
git checkout timetiling_on_fwi
```
Set env variables and pinning.
```
DEVITO_LANGUAGE=openmp
OMP_PROC_BIND=cores #(avoid hyperthreading)
DEVITO_LOGGING=DEBUG
```

For the isotropic acoustic case:

`DEVITO_JIT_BACKDOOR=0 python3 examples/seismic/acoustic/acoustic_example.py -so 4 -d 512 512 512 --tn 512` 

For the isotropic elastic case:

`DEVITO_JIT_BACKDOOR=0 python3 examples/seismic/elastic/demo_elastic3.py -so 4 -d 512 512 512 --tn 512` 

For the acoustic case:

`DEVITO_JIT_BACKDOOR=0 python3 examples/seismic/tti/tti_example.py -so 4 -d 512 512 512 --tn 512` 


This will generate code for a space order 4 acoustic devito kernel and another kernel that we will modify.

The generated log will end executing a kernel under `===Temporal blocking================================`

```
===Temporal blocking======================================
Allocating memory for n(1,)
Allocating memory for usol(3, 236, 236, 236)
gcc -O3 -g -fPIC -Wall -std=c99 -march=native -Wno-unused-result -Wno-unused-variable -Wno-unused-but-set-variable -ffast-math -shared -fopenmp /tmp/devito-jitcache-uid1000/xxx-hash-xxx.c -lm -o /tmp/devito-jitcache-uid1000/xxx-hash-xxx.so
Operator `Kernel` jit-compiled `/tmp/devito-jitcache-uid1000/xxx-hash-xxx.c` in 0.48 s with `GNUCompiler`
Operator `Kernel` run in 0.49 s
```

Each example, (acoustic/elastic/tti) has a folder named `kernels`

Copy the matching hash kernel from `devito/examples/seismic/acoustic/kernels/` to the `tmp/devito-jit-cache..../xxx-hash-xxx.c`
`cp kernels/kernel_so8_acoustic.c /tmp/devito-jitcache-uid1000/d3cee7726ce639b303537a387aa51cd42d9feecd.c`

Then try `DEVITO_JIT_BACKDOOR=1 python3 examples/seismic/acoustic/demo_temporal_sources.py -so 4`
to re-run. Now the norms should match. If not contact me ASAP :-).

Use arguments `-d nx ny nx` , `-tn timesteps` to pass domain size and number of timesteps.
e.g.:

`DEVITO_JIT_BACKDOOR=1 python3 examples/seismic/acoustic/acoustic_example.py -d 200 200 200 --tn 100 -so 8`

There exist available kernels for space orders 4, 8, 12.

# Tuning Devito
Run Devito with `DEVITO_LOGGING=aggressive` so as to ensure that one of the best space-blocking configurations are selected.

# Tuning time-tiled kernel
In order to manually tune the Devito time-tiled kernel we provide code to parametrize runs across combinations of tile/blocks.

Note:

`xb_size, yb_size == tiles`
`x0_blk0_size, y0_blk0_size == blocks`

As of YASK:
`From YASK: Although the terms "block" and "tile" are often used interchangeably, in
this section, we [arbitrarily] use the term "block" for spatial-only grouping
and "tile" when multiple temporal updates are allowed.`

Let me know your findings and your performance results. https://opesci-slackin.now.sh/ at #time-tiling

Depending on platform, I would expect speed-ups along the results presented in this paper: https://arxiv.org/abs/2010.10248
