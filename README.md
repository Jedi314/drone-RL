# Quadcopter control based on deep reinforcement learning
I use [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones/).

**Installation**
```sh
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones/

conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

```

**How to use**

For train:
```sh
python3 train_racing.py
```
For test:
```sh
python3 test_racing.py
```
