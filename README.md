# robot_planning

The overall structure of the repo is illustrated in the following diagram,
![Robot_planning structure](https://user-images.githubusercontent.com/26529114/129230641-7c27386a-40d4-4fab-94a8-92d30b4673be.png)
with the decentralized controller,
![Robot_planning decentrailized controller structure](https://user-images.githubusercontent.com/26529114/129230638-f1b36637-ddf8-46ee-a78d-0952edf4e2b9.png)


## Setup

### Install prerequisites
In terminal, change directory to `robot_planning` and run `pip install -r requirements.txt`. Pytorch and Mosek Fusion will need to be installed manually.
1. Install Mosek `pip install Mosek`. Obtain a personal academic license and place it in the required folder https://docs.mosek.com/9.3/licensing/quickstart.html#i-don-t-have-a-license-file-yet
2. Intall pytorch with the correct options selected for your machine https://pytorch.org/get-started/locally/

### Install the robot_planning repo
Run the following 3 commands to install the repo:
1. `cp setup.py ..`
2. `cd ..`
3. `pip install -e .`


## Editing this repository

1. Create your own branch from master and name it firstname-lastname
2. Only push directly to your personal branch
3. Create new tests when you add a new feature or script, and ensure that all tests still pass with your changes
3. When a branch is stable and changes are ready to be shared, create a pull request on the master branch for the team to review
4. Only merge stable branches into master after checking all tests are passing; never push commits directly to master

## Running the code

1. Change directory to `robot_planning/scripts`
2. Run the desired script (eg `python3 run_Autorally_CSSMPC.py`)
3. The run configuration is defined in the `configs/` folder with the corresponding `.cfg` file

## Running tests

1. Tests are located in the `tests/` directory
2. They can be run from the pycharm GUI by right clicking on the `tests` folder and selecting `run pytest in tests` or from the terminal by changing directory to `tests` and then running `pytest`
