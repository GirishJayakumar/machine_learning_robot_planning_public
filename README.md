# robot_planning
## Install required packages
In terminal, cd to `robot_planning` and run `pip install -r requirements.txt`. Pytorch and Mosek Fusion will need to be installed manually.
Copy `setup.py` outside of the `robot_planning` directory and then from the parent directory run `pip install -e .`. That is, run the following commands 3 commands: `cp setup.py ..`, `cd ..`, `pip install -e .`.

The overall structure of the repo is illustrated in the following diagram,
![Robot_planning structure](https://user-images.githubusercontent.com/26529114/129230641-7c27386a-40d4-4fab-94a8-92d30b4673be.png)
with the decentralized controller,
![Robot_planning decentrailized controller structure](https://user-images.githubusercontent.com/26529114/129230638-f1b36637-ddf8-46ee-a78d-0952edf4e2b9.png)

