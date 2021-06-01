# CSC3022F Machine Learning Assignment 2: Reinforcement Learning

ValueIteration.py implements the algorithm of value iteration.

QLearning.py implements the algorithm of Q learning.

Helper.py provides a helper function to extract an optimal policy based on the
passed record.

The requirements.txt provides a list of packages that are required for setting
up the virtual environment.

A makefile has been provided to automate the building of the virtual env and the
removal the compiled files.

```ssh
make
```
creates the virtual environment and installs the necessary package in the
virtual environment for this project.

```ssh
make clean
```
removes the virtual environment and gifs and compilation files.

In order to run the scripts, use the following commands:

```ssh
source venv/bin/activate
```
activates the virtual environment

```ssh
python3 ValueIteration.py <width> <height> [options]
```
runs the value iteration algorithm with an environment of sizes specified by
<width> and <height>

Available options:  -start xpos ypos: starting state, random if not given

                    -end xpos ypos: end state, random if not given

                    -k num: number of landmines, 3 by default

                    -gamma g: discount factor, 0.8 by default

```ssh
python3 QLearning.py <width> <height> [options]
```
runs the Q learning algorithm with an environment of sizes specified by
<width> and <height>

Available options:  -start xpos ypos: starting state, random if not given

                    -end xpos ypos: end state, random if not given

                    -k num: number of landmines, 3 by default

                    -gamma g: discount factor, 0.8 by default

                    -learning l: learning rate, decaying learning rate if not given

                    -epochs e: number of epochs the agent will learn for, 500 by default
