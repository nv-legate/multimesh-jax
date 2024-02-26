# Running PaxML with Legate-Jax

The workflow for running large tests with PaxML should generally consist of a few steps.
The parameters `<...>` passed to each step should be exactly the same, including the `--gpus` and `--nodes` flags.
For a full list of parameters, you can run:

```
$ ./run.py <...> --help
```

## Run a test job to generate the HLO module for the training step locally

```
$ ./run.py <...> --dump=<folder> --dump-only
```

On success, this should generate an HLO module binary proto for the training step
in a file `module_XXXX_pjit_autoshard_step_fn.before_optimizations.hlo.pb` in the given folder.


## Run a compiler validation of the HLO module to ensure that Legate-Jax is able to partition and compile the module locally

```
$ ./run.py <...> --hlo=<hlo>
```

On success, this should print a series of outputs showing all the HLO modules that will be executed
along with their memory and compute requirements:

```
Buffer Summary for Module layers_0
    Input   : 0.27419GB
      No. input buffers    = 18
    Output  : 0.247988GB
      No. output buffers   = 9
    Temp    : 0.327994GB
      No. temp buffers     = 1
    Constant: 1.2e-08GB
      No. constant buffers = 3
    Total   : 0.850172GB
   40 total buffers
```

## Run full PaxML on the desired platform

Based on the compile output, parameters may need to be tuned - particularly the total amount of memory and the eager allocation.

```
$ srun -n 4 -N 4 ./run.py <...>
```
