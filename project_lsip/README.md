# Instructions for running the code
The project tries to re-produce the paper "Learning Fast Optimizers for Contextual Stochastic Integer Programs" (http://auai.org/uai2018/proceedings/papers/217.pdf).

## Install the requirements
- Python (3.5 or above)
- Pytorch (1.0.1)
- SCIP Suite (6.0.1)
- pyscipopt

### Instructions for installing `SCIP Suite` and `pyscipopt`

1. Install the latest cmake 

```
conda install cmake
hash -r
cmake --version
```

2. Download the latest SCIP suite source code in your local workspace and untar it.

```
cd /local_workspace/<username>
tar -xzf scipoptsuite-6.0.1.tgz /local_workspace/<username>/scipoptsuite-6.0.1
```

3. In this folder, edit `/local_workspace/<username>/scipoptsuite-6.0.1/CMakeLists.txt` and delete the lines 16-18:

```
if(${BISON_FOUND} AND ${FLEX_FOUND} AND ${GMP_FOUND})
  add_subdirectory(zimpl)
endif()
```
This is a hack to prevent cmake from finding ZIMPL, which is not compatible with the latest gcc.

4. Create a build folder and run cmake, make and make install with the desired install directory, e.g. /local_workspace/&lt;username>/scipoptsuite.

    ```
    cd /local_workspace/<username>/scipoptsuite-6.0.1
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=/local_workspace/<username>/scipoptsuite/
    make INSTALLDIR=/local_workspace/<username>/scipoptsuite/
    make install INSTALLDIR=/local_workspace/<username>/scipoptsuite/
    ```

    You can now remove the source code tar and folder if desired, as they are no longer needed.

5. In your bashrc file, add environment variables for pyscipopt.

    ```
    export SCIPOPTDIR="/local_workspace/<username>/scipoptsuite"
    alias scip='/local_workspace/<username>/scipoptsuite/bin/scip'
    ```
    `SCIPOPTDIR` needs to have a subdirectory `lib` that contains the
    library, e.g. `libscip.so` (for Linux) and a subdirectory `include` that
    contains the corresponding header files:

    ```
    SCIPOPTDIR
      > lib
        > libscip.so ...
      > include
        > scip
        > lpi
        > nlpi
        > ...
    ```
    
6. Install pyscipopt by running `pip install pyscipopt`.

## Running the code

`python main.py <options>`

The following options are available. 

- `--init_model`: Initialisation model to be used (NADE, LSTM or NN)
- `--init_lr_rate`: Learning rate for initialisation policy
- `--epochs`: Number of epochs to train the policy
- `--init_lr_rate`: Learning rate for the initialisation Policy
- `--use_baseline`: Indicator to use baseline network during training initialisation policy
- `--num_of_scenarios_for_expectation`: Number of scenarios to generate in the environment to evaluate the expectation
- `--problem`: The class of problem we are solving. (Knapsack by default)
- `--dim_hidden`: Size of the hidden layer of network
- `--dim_problem`: Size of the problem
- `--window_size`: Number of past-time steps to consider while defining the state
- `--num_of_scenarios_in_state`: NUmber of scenarios to to consider to define the state
- `--train_mode`: How would we like to train i.e. training the policies individually or jointly (INIT, LOCAL or JOINT)
- `--gamma`: Discount factor
- `--beta`: Scalar multiplier for entropy
- `--num_local_move`: Number of local moves the agent can make to improve the solution
- `--lr_a2c`: Learning rate for the A2C based Local-Move Policy 

## Credits
We are extremely grateful to [@dchetelat](https://github.com/dchetelat) for devoting his time to help us setup 
the SCIP environment (without which this project would 
not have been started) and solving our doubts.
