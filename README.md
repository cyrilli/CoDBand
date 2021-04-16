# Code for "When and Whom to Collaborate with in a Changing Environment: A Collaborative Dynamic Bandit Solution"

This repository contains implementation of the proposed algorithm CoDBand, and baseline algorithms for comparison:
- LinUCB
- dLinUCB and adTS
- CLUB and SCLUB
- oracleLinUCB

For experiments on the synthetic dataset, run:
```console
python Simulation.py --T 3000 --SMIN 500 --SMAX 3000 --n 10 --m 2 --sigma 0.1  # bandit parameters are uniformly sampled from a set of size m
python SimulationDP.py --T 3000 --SMIN 500 --SMAX 3000 --n 10 --m 2 --sigma 0.1  # bandit parameters are generated according to Chinese restaurant process
```
To experiment with different environment settings, specify parameters:
- T: number of iterations to run
- SMIN: minimum length of each stationary period
- SMAX: maximum length of each stationary period
- n: number of users
- m: number of unique parameters shared by users
- sigma: standard deviation of Gaussian noise in observed reward

Detailed description of how the simulation environment works can be found in Section 5.1 of the paper.

Experiment results can be found in "./SimulationResults/" folder, which contains:
- "[namelabel]\_[startTime].png": plot of accumulated regret over iteration for each algorithm
- "[namelabel]\_AccRegret\_[startTime].csv": regret at each iteration for each algorithm
- "[namelabel]\_ParameterEstimation\_[startTime].csv": l2 norm between estimated and ground-truth parameter at each iteration for each algorithm
- "Config\_[startTime].json": stores hyper parameters of all algorithms for this experiment

For experiments on realworld dataset, e.g. LastFM, Delicious, MovieLens, 
- First download these publicly available data into Dataset folder 
- Then process the dataset following instructions in Section 5.2 of the paper, which would generate the item feature vector file and the event file. Example scripts for processing data are given in Dataset folder.
- Run experiments using the provided python script "DeliciousLastFMAndMovieLens.py"
